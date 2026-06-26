from __future__ import annotations

from itertools import combinations
from pathlib import Path

from kargo_media_recommender.benchmarks import calculate_benchmarks, estimate_impressions
from kargo_media_recommender.data import load_media_data
from kargo_media_recommender.display import (
    format_currency,
    format_impressions,
    format_percent,
)
from kargo_media_recommender.schemas import (
    ClientRequirements,
    KPI_LABELS,
    KPI_SHORT_LABELS,
    MediaData,
    ProductCandidate,
    RecommendationResult,
    RecommendedProduct,
    RejectedProduct,
)


class RecommendationEngine:
    _BUDGET_EPSILON = 0.01
    _IMPRESSION_EPSILON = 0.5

    def __init__(self, media_data: MediaData, max_bundle_size: int = 3) -> None:
        if max_bundle_size < 2:
            raise ValueError("max_bundle_size must be at least 2.")
        self.media_data = media_data
        self.max_bundle_size = max_bundle_size
        self.benchmarks = calculate_benchmarks(media_data.history_rows)

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path | None = None,
        max_bundle_size: int = 3,
    ) -> "RecommendationEngine":
        return cls(load_media_data(data_dir), max_bundle_size=max_bundle_size)

    def recommend(self, requirements: ClientRequirements) -> RecommendationResult:
        missing_fields = requirements.missing_required_fields
        if missing_fields:
            return RecommendationResult(
                status="clarification_needed",
                requirements=requirements,
                missing_fields=missing_fields,
                recommended_products=(),
                rejected_alternatives=(),
                candidates=(),
                rationale="Required campaign details are missing before a recommendation can be calculated.",
                tradeoffs=(),
            )

        candidates = self._build_candidates(requirements)
        if not candidates:
            return RecommendationResult(
                status="no_viable_option",
                requirements=requirements,
                missing_fields=(),
                recommended_products=(),
                rejected_alternatives=(),
                candidates=(),
                rationale=(
                    "No product data matches the requested vertical and geo, so a recommendation "
                    "cannot be calculated from the available catalog."
                ),
                tradeoffs=(),
            )

        sorted_candidates = self._sort_candidates(candidates, requirements.primary_kpi)
        kpi_sorted_candidates = self._sort_candidates_by_kpi(candidates, requirements.primary_kpi)
        viable = [candidate for candidate in sorted_candidates if candidate.viable]
        selected = viable[0] if viable else None
        best_single = selected or (kpi_sorted_candidates[0] if kpi_sorted_candidates else None)
        bundle = self._build_bundle(kpi_sorted_candidates, requirements)

        if requirements.recommendation_style == "single_product_preferred" and best_single:
            return self._single_product_result(
                best_single,
                requirements,
                sorted_candidates,
                budget_caveat=not best_single.viable,
            )

        if bundle and requirements.recommendation_style == "maximize_budget_delivery":
            selected_ids = {product.product_id for product in bundle}
            rejected = self._rejected_alternatives(
                sorted_candidates,
                selected_ids=selected_ids,
                primary_kpi=requirements.primary_kpi,
                selected_plan_kpi=self._plan_kpi(bundle, requirements.primary_kpi),
                selected_plan_label="selected bundle",
                best_single_product_id=selected.product.product_id if selected else None,
            )
            return RecommendationResult(
                status="bundle",
                requirements=requirements,
                missing_fields=(),
                recommended_products=tuple(bundle),
                rejected_alternatives=rejected,
                candidates=tuple(kpi_sorted_candidates),
                rationale=self._bundle_rationale(bundle, requirements),
                tradeoffs=self._bundle_tradeoffs(bundle, requirements, selected),
            )

        if bundle and self._bundle_beats_single(bundle, selected, requirements.primary_kpi):
            selected_ids = {product.product_id for product in bundle}
            rejected = self._rejected_alternatives(
                sorted_candidates,
                selected_ids=selected_ids,
                primary_kpi=requirements.primary_kpi,
                selected_plan_kpi=self._plan_kpi(bundle, requirements.primary_kpi),
                selected_plan_label="selected bundle",
                best_single_product_id=selected.product.product_id if selected else None,
            )
            return RecommendationResult(
                status="bundle",
                requirements=requirements,
                missing_fields=(),
                recommended_products=tuple(bundle),
                rejected_alternatives=rejected,
                candidates=tuple(kpi_sorted_candidates),
                rationale=self._bundle_rationale(bundle, requirements),
                tradeoffs=self._bundle_tradeoffs(bundle, requirements, selected),
            )

        if selected:
            return self._single_product_result(
                selected,
                requirements,
                sorted_candidates,
                budget_caveat=False,
            )

        if bundle:
            selected_ids = {product.product_id for product in bundle}
            rejected = self._rejected_alternatives(
                sorted_candidates,
                selected_ids=selected_ids,
                primary_kpi=requirements.primary_kpi,
                selected_plan_kpi=self._plan_kpi(bundle, requirements.primary_kpi),
                selected_plan_label="selected bundle",
            )
            return RecommendationResult(
                status="bundle",
                requirements=requirements,
                missing_fields=(),
                recommended_products=tuple(bundle),
                rejected_alternatives=rejected,
                candidates=tuple(kpi_sorted_candidates),
                rationale=self._bundle_rationale(bundle, requirements),
                tradeoffs=(
                    "No single product could satisfy the full budget and scale constraints, so spend was allocated across the highest-ranked products with available capacity.",
                ),
            )

        if best_single:
            return self._single_product_result(
                best_single,
                requirements,
                sorted_candidates,
                budget_caveat=True,
            )

        return RecommendationResult(
            status="no_viable_option",
            requirements=requirements,
            missing_fields=(),
            recommended_products=(),
            rejected_alternatives=self._rejected_alternatives(
                sorted_candidates,
                selected_ids=set(),
                primary_kpi=requirements.primary_kpi,
                limit=len(sorted_candidates),
            ),
            candidates=tuple(sorted_candidates),
            rationale="No product or bundle can satisfy the budget, KPI, geo, and scale constraints with usable inventory.",
            tradeoffs=(
                "Consider lowering the budget, lowering the impression goal, widening the geo, or accepting lower inventory confidence.",
            ),
        )

    def _build_candidates(self, requirements: ClientRequirements) -> list[ProductCandidate]:
        assert requirements.vertical is not None
        assert requirements.geo is not None
        assert requirements.budget is not None

        candidates: list[ProductCandidate] = []
        for product in sorted(self.media_data.products.values(), key=lambda item: item.product_id):
            benchmark = self.benchmarks.get((product.product_id, requirements.vertical))
            inventory = self.media_data.inventory.get(
                (product.product_id, requirements.vertical, requirements.geo)
            )
            if benchmark is None or inventory is None:
                continue

            forecasted_impressions = estimate_impressions(requirements.budget, product.cpm)
            reasons = self._rejection_reasons(
                forecasted_impressions=forecasted_impressions,
                confidence_adjusted_available=inventory.confidence_adjusted_available_imps,
                impression_goal=requirements.impression_goal,
            )
            candidates.append(
                ProductCandidate(
                    product=product,
                    benchmark=benchmark,
                    inventory=inventory,
                    estimated_impressions=forecasted_impressions,
                    rejection_reasons=tuple(reasons),
                )
            )

        return candidates

    @staticmethod
    def _rejection_reasons(
        forecasted_impressions: float,
        confidence_adjusted_available: float,
        impression_goal: int | None,
    ) -> list[str]:
        reasons: list[str] = []
        if confidence_adjusted_available < forecasted_impressions:
            reasons.append(
                "Insufficient usable inventory for the full budget "
                f"({format_impressions(confidence_adjusted_available)} usable vs. "
                f"{format_impressions(forecasted_impressions)} needed)."
            )
        if impression_goal is not None and forecasted_impressions < impression_goal:
            reasons.append(
                f"Budget forecasts {format_impressions(forecasted_impressions)} impressions, "
                f"below the {format_impressions(impression_goal)} impression goal."
            )
        if impression_goal is not None and confidence_adjusted_available < impression_goal:
            reasons.append(
                "Usable inventory is below the impression goal "
                f"({format_impressions(confidence_adjusted_available)} usable vs. "
                f"{format_impressions(impression_goal)} needed)."
            )
        return reasons

    @staticmethod
    def _sort_candidates_by_kpi(
        candidates: list[ProductCandidate],
        primary_kpi: str,
    ) -> list[ProductCandidate]:
        return sorted(
            candidates,
            key=lambda candidate: (
                -candidate.score(primary_kpi),
                -candidate.inventory.inventory_confidence,
                -candidate.product.cpm,
                candidate.product.product_id,
            ),
        )

    @staticmethod
    def _sort_candidates(
        candidates: list[ProductCandidate],
        primary_kpi: str,
    ) -> list[ProductCandidate]:
        return sorted(
            candidates,
            key=lambda candidate: (
                not candidate.viable,
                -candidate.score(primary_kpi),
                -candidate.inventory.inventory_confidence,
                -candidate.estimated_impressions,
                candidate.product.cpm,
                candidate.product.product_id,
            ),
        )

    def _build_bundle(
        self,
        candidates: list[ProductCandidate],
        requirements: ClientRequirements,
    ) -> list[RecommendedProduct]:
        assert requirements.budget is not None

        best_bundle: list[RecommendedProduct] = []
        best_key: tuple | None = None
        max_bundle_size = min(self.max_bundle_size, len(candidates))

        for bundle_size in range(2, max_bundle_size + 1):
            for subset in combinations(candidates, bundle_size):
                for allocation in self._candidate_allocations(subset, requirements):
                    bundle = self._bundle_from_allocation(subset, allocation)
                    if not self._bundle_satisfies_constraints(bundle, requirements):
                        continue

                    key = self._bundle_rank_key(
                        bundle,
                        requirements.primary_kpi,
                    )
                    if best_key is None or key > best_key:
                        best_bundle = bundle
                        best_key = key

        return best_bundle

    def _candidate_allocations(
        self,
        candidates: tuple[ProductCandidate, ...],
        requirements: ClientRequirements,
    ) -> list[tuple[float, ...]]:
        assert requirements.budget is not None

        total_budget = requirements.budget
        spend_caps = [
            min(self._spend_capacity(candidate), total_budget) for candidate in candidates
        ]
        if sum(spend_caps) + self._BUDGET_EPSILON < total_budget:
            return []

        impression_rates = [1000 / candidate.product.cpm for candidate in candidates]
        constraints = self._allocation_boundary_constraints(
            spend_caps,
            impression_rates,
            requirements.impression_goal,
        )
        allocations: list[tuple[float, ...]] = []
        seen: set[tuple[float, ...]] = set()

        for active_constraints in combinations(constraints, len(candidates) - 1):
            equations = [([1.0] * len(candidates), total_budget), *active_constraints]
            allocation = self._solve_linear_system(equations)
            if allocation is None:
                continue

            normalized = self._normalize_allocation(allocation)
            if not self._allocation_is_feasible(
                normalized,
                spend_caps,
                impression_rates,
                requirements.impression_goal,
                total_budget,
            ):
                continue

            rounded = tuple(round(value, 6) for value in normalized)
            if rounded in seen:
                continue
            seen.add(rounded)
            allocations.append(normalized)

        return allocations

    @staticmethod
    def _allocation_boundary_constraints(
        spend_caps: list[float],
        impression_rates: list[float],
        impression_goal: int | None,
    ) -> list[tuple[list[float], float]]:
        constraints: list[tuple[list[float], float]] = []
        for index, spend_cap in enumerate(spend_caps):
            lower = [0.0] * len(spend_caps)
            lower[index] = 1.0
            constraints.append((lower, 0.0))

            upper = [0.0] * len(spend_caps)
            upper[index] = 1.0
            constraints.append((upper, spend_cap))

        if impression_goal is not None:
            constraints.append((list(impression_rates), float(impression_goal)))

        return constraints

    @classmethod
    def _allocation_is_feasible(
        cls,
        allocation: tuple[float, ...],
        spend_caps: list[float],
        impression_rates: list[float],
        impression_goal: int | None,
        total_budget: float,
    ) -> bool:
        if abs(sum(allocation) - total_budget) > cls._BUDGET_EPSILON:
            return False

        for budget, spend_cap in zip(allocation, spend_caps, strict=True):
            if budget < -cls._BUDGET_EPSILON:
                return False
            if budget - spend_cap > cls._BUDGET_EPSILON:
                return False

        if impression_goal is not None:
            impressions = sum(
                budget * impression_rate
                for budget, impression_rate in zip(allocation, impression_rates, strict=True)
            )
            if impressions + cls._IMPRESSION_EPSILON < impression_goal:
                return False

        positive_allocations = [
            budget for budget in allocation if budget > cls._BUDGET_EPSILON
        ]
        return len(positive_allocations) >= 2

    @staticmethod
    def _normalize_allocation(allocation: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(0.0 if abs(value) < 1e-7 else value for value in allocation)

    @staticmethod
    def _solve_linear_system(
        equations: list[tuple[list[float], float]],
    ) -> tuple[float, ...] | None:
        size = len(equations)
        matrix = [
            [*coefficients, rhs]
            for coefficients, rhs in equations
        ]

        for pivot_index in range(size):
            pivot_row = max(
                range(pivot_index, size),
                key=lambda row_index: abs(matrix[row_index][pivot_index]),
            )
            if abs(matrix[pivot_row][pivot_index]) < 1e-9:
                return None
            matrix[pivot_index], matrix[pivot_row] = matrix[pivot_row], matrix[pivot_index]

            pivot = matrix[pivot_index][pivot_index]
            matrix[pivot_index] = [value / pivot for value in matrix[pivot_index]]

            for row_index in range(size):
                if row_index == pivot_index:
                    continue
                factor = matrix[row_index][pivot_index]
                matrix[row_index] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(
                        matrix[row_index],
                        matrix[pivot_index],
                        strict=True,
                    )
                ]

        return tuple(row[-1] for row in matrix)

    def _bundle_from_allocation(
        self,
        candidates: tuple[ProductCandidate, ...],
        allocation: tuple[float, ...],
    ) -> list[RecommendedProduct]:
        bundle: list[RecommendedProduct] = []
        for candidate, budget in zip(candidates, allocation, strict=True):
            if budget <= self._BUDGET_EPSILON:
                continue
            forecasted_impressions = estimate_impressions(budget, candidate.product.cpm)
            bundle.append(
                self._recommended_from_candidate(
                    candidate,
                    budget=budget,
                    forecasted_impressions=forecasted_impressions,
                )
            )
        return bundle

    @classmethod
    def _bundle_satisfies_constraints(
        cls,
        bundle: list[RecommendedProduct],
        requirements: ClientRequirements,
    ) -> bool:
        assert requirements.budget is not None

        if len(bundle) < 2:
            return False

        total_budget = sum(product.budget for product in bundle)
        if abs(total_budget - requirements.budget) > cls._BUDGET_EPSILON:
            return False

        total_impressions = sum(product.forecasted_impressions for product in bundle)
        if (
            requirements.impression_goal is not None
            and total_impressions + cls._IMPRESSION_EPSILON < requirements.impression_goal
        ):
            return False

        for product in bundle:
            if (
                product.forecasted_impressions
                - product.confidence_adjusted_available_impressions
                > cls._IMPRESSION_EPSILON
            ):
                return False

        return True

    @classmethod
    def _bundle_rank_key(
        cls,
        bundle: list[RecommendedProduct],
        primary_kpi: str,
    ) -> tuple:
        total_impressions = sum(product.forecasted_impressions for product in bundle)
        weighted_confidence = sum(
            product.forecasted_impressions * product.inventory_confidence
            for product in bundle
        ) / total_impressions
        product_names = "|".join(product.product_name for product in bundle)
        return (
            cls._spend_weighted_plan_kpi(bundle, primary_kpi),
            cls._plan_kpi(bundle, primary_kpi),
            total_impressions,
            weighted_confidence,
            -len(bundle),
            product_names,
        )

    @staticmethod
    def _spend_weighted_plan_kpi(
        products: list[RecommendedProduct],
        primary_kpi: str,
    ) -> float:
        total_budget = sum(product.budget for product in products)
        if total_budget <= 0:
            return 0

        weighted_kpi = 0.0
        for product in products:
            if primary_kpi == "ctr":
                kpi_value = product.benchmark_ctr
            elif primary_kpi == "in_view_rate":
                kpi_value = product.benchmark_in_view_rate
            else:
                raise ValueError(f"Unsupported primary KPI: {primary_kpi!r}")
            weighted_kpi += product.budget * kpi_value

        return weighted_kpi / total_budget

    @classmethod
    def _bundle_beats_single(
        cls,
        bundle: list[RecommendedProduct],
        selected: ProductCandidate | None,
        primary_kpi: str,
    ) -> bool:
        if not bundle:
            return False
        if selected is None:
            return True
        return cls._plan_kpi(bundle, primary_kpi) > selected.score(primary_kpi)

    @staticmethod
    def _plan_kpi(products: list[RecommendedProduct], primary_kpi: str) -> float:
        total_impressions = sum(product.forecasted_impressions for product in products)
        if total_impressions <= 0:
            return 0

        weighted_kpi = 0.0
        for product in products:
            if primary_kpi == "ctr":
                kpi_value = product.benchmark_ctr
            elif primary_kpi == "in_view_rate":
                kpi_value = product.benchmark_in_view_rate
            else:
                raise ValueError(f"Unsupported primary KPI: {primary_kpi!r}")
            weighted_kpi += product.forecasted_impressions * kpi_value

        return weighted_kpi / total_impressions

    @staticmethod
    def _recommended_from_candidate(
        candidate: ProductCandidate,
        budget: float,
        forecasted_impressions: float,
    ) -> RecommendedProduct:
        return RecommendedProduct(
            product_id=candidate.product.product_id,
            product_name=candidate.product.product_name,
            budget=budget,
            cpm=candidate.product.cpm,
            forecasted_impressions=forecasted_impressions,
            benchmark_ctr=candidate.benchmark.ctr,
            benchmark_in_view_rate=candidate.benchmark.in_view_rate,
            available_impressions=candidate.inventory.available_imps,
            inventory_confidence=candidate.inventory.inventory_confidence,
            confidence_adjusted_available_impressions=(
                candidate.inventory.confidence_adjusted_available_imps
            ),
        )

    def _single_product_result(
        self,
        selected: ProductCandidate,
        requirements: ClientRequirements,
        candidates: list[ProductCandidate],
        budget_caveat: bool,
    ) -> RecommendationResult:
        assert requirements.budget is not None
        assert requirements.primary_kpi is not None

        allocated_budget = self._single_product_budget(selected, requirements)
        forecasted_impressions = estimate_impressions(allocated_budget, selected.product.cpm)
        recommended = self._recommended_from_candidate(
            selected,
            budget=allocated_budget,
            forecasted_impressions=forecasted_impressions,
        )
        selected_ids = {selected.product.product_id}
        rejected = self._rejected_alternatives(
            candidates,
            selected_ids=selected_ids,
            primary_kpi=requirements.primary_kpi,
            selected_plan_kpi=selected.score(requirements.primary_kpi),
            selected_plan_label="selected product",
        )

        return RecommendationResult(
            status="single_product_budget_caveat" if budget_caveat else "single_product",
            requirements=requirements,
            missing_fields=(),
            recommended_products=(recommended,),
            rejected_alternatives=rejected,
            candidates=tuple(candidates),
            rationale=(
                self._single_product_budget_caveat_rationale(
                    selected,
                    recommended,
                    requirements,
                )
                if budget_caveat
                else self._single_product_rationale(selected, requirements)
            ),
            tradeoffs=self._tradeoffs(candidates, selected_ids=selected_ids),
        )

    @classmethod
    def _single_product_budget(
        cls,
        candidate: ProductCandidate,
        requirements: ClientRequirements,
    ) -> float:
        assert requirements.budget is not None

        spend_capacity = cls._spend_capacity(candidate)
        return min(requirements.budget, spend_capacity)

    @staticmethod
    def _spend_capacity(candidate: ProductCandidate) -> float:
        return (
            candidate.inventory.confidence_adjusted_available_imps
            / 1000
            * candidate.product.cpm
        )

    def _rejected_alternatives(
        self,
        candidates: list[ProductCandidate],
        selected_ids: set[str],
        primary_kpi: str,
        limit: int = 3,
        best_single_product_id: str | None = None,
        selected_plan_kpi: float | None = None,
        selected_plan_label: str = "selected recommendation",
    ) -> tuple[RejectedProduct, ...]:
        rejected: list[RejectedProduct] = []
        for candidate in candidates:
            if candidate.product.product_id in selected_ids:
                continue

            reasons = self._display_rejection_reasons(
                candidate,
                primary_kpi=primary_kpi,
                selected_plan_kpi=selected_plan_kpi,
                selected_plan_label=selected_plan_label,
                best_single_product_id=best_single_product_id,
            )

            rejected.append(
                RejectedProduct(
                    product_id=candidate.product.product_id,
                    product_name=candidate.product.product_name,
                    benchmark_ctr=candidate.benchmark.ctr,
                    benchmark_in_view_rate=candidate.benchmark.in_view_rate,
                    estimated_impressions=candidate.estimated_impressions,
                    reasons=tuple(reasons),
                )
            )

            if len(rejected) >= limit:
                break

        return tuple(rejected)

    @staticmethod
    def _display_rejection_reasons(
        candidate: ProductCandidate,
        primary_kpi: str,
        selected_plan_kpi: float | None,
        selected_plan_label: str,
        best_single_product_id: str | None = None,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        comparison_reason = RecommendationEngine._kpi_comparison_reason(
            candidate,
            primary_kpi=primary_kpi,
            selected_plan_kpi=selected_plan_kpi,
            selected_plan_label=selected_plan_label,
            best_single_product_id=best_single_product_id,
        )
        if comparison_reason:
            if selected_plan_label == "selected bundle":
                return (comparison_reason,)
            reasons.append(comparison_reason)

        reasons.extend(candidate.rejection_reasons)
        if reasons:
            return tuple(reasons)

        fallback_reason = RecommendationEngine._fallback_rejection_reason(
            candidate,
            primary_kpi=primary_kpi,
            selected_plan_kpi=selected_plan_kpi,
            selected_plan_label=selected_plan_label,
            best_single_product_id=best_single_product_id,
        )
        return (fallback_reason,)

    @staticmethod
    def _kpi_comparison_reason(
        candidate: ProductCandidate,
        primary_kpi: str,
        selected_plan_kpi: float | None,
        selected_plan_label: str,
        best_single_product_id: str | None,
    ) -> str | None:
        if selected_plan_kpi is None:
            return None

        candidate_kpi = candidate.score(primary_kpi)
        if candidate_kpi >= selected_plan_kpi - 1e-12:
            return None

        kpi_name = KPI_LABELS[primary_kpi]
        comparison = f"({format_percent(candidate_kpi)} vs. {format_percent(selected_plan_kpi)})."
        if candidate.product.product_id == best_single_product_id:
            return (
                f"Best single-product alternative, but lower {kpi_name} than the "
                f"{selected_plan_label} {comparison}"
            )
        return f"Lower {kpi_name} than the {selected_plan_label} {comparison}"

    @staticmethod
    def _fallback_rejection_reason(
        candidate: ProductCandidate,
        primary_kpi: str,
        selected_plan_kpi: float | None,
        selected_plan_label: str,
        best_single_product_id: str | None,
    ) -> str:
        kpi_name = KPI_LABELS[primary_kpi]
        if selected_plan_kpi is None:
            return "Not selected."

        candidate_kpi = candidate.score(primary_kpi)
        comparison = f"({format_percent(candidate_kpi)} vs. {format_percent(selected_plan_kpi)})."
        if candidate.product.product_id == best_single_product_id:
            return (
                f"Best single-product alternative, but it did not improve {kpi_name} "
                f"over the {selected_plan_label} {comparison}"
            )
        return (
            f"Viable, but did not improve {kpi_name} over the "
            f"{selected_plan_label} {comparison}"
        )

    @staticmethod
    def _tradeoffs(
        candidates: list[ProductCandidate],
        selected_ids: set[str],
    ) -> tuple[str, ...]:
        selected_rejections = [
            candidate
            for candidate in candidates
            if candidate.product.product_id not in selected_ids and candidate.rejection_reasons
        ]
        tradeoffs = []
        if selected_rejections:
            top_rejected = selected_rejections[0]
            reason = " ".join(top_rejected.rejection_reasons).rstrip(".")
            reason = reason[:1].lower() + reason[1:]
            tradeoffs.append(
                f"{top_rejected.product.product_name} was not selected because {reason}."
            )
        return tuple(tradeoffs)

    @classmethod
    def _bundle_tradeoffs(
        cls,
        bundle: list[RecommendedProduct],
        requirements: ClientRequirements,
        selected: ProductCandidate | None,
    ) -> tuple[str, ...]:
        assert requirements.primary_kpi is not None

        kpi_short_name = KPI_SHORT_LABELS[requirements.primary_kpi]
        bundle_kpi = cls._plan_kpi(bundle, requirements.primary_kpi)
        product_list = " + ".join(product.product_name for product in bundle)
        total_impressions = sum(product.forecasted_impressions for product in bundle)
        tradeoffs = [
            f"Blended {kpi_short_name}: {format_percent(bundle_kpi)}.",
            f"Budget used: {format_currency(requirements.budget)} across {product_list}.",
        ]

        if requirements.impression_goal is not None:
            tradeoffs.append(
                "Forecasted delivery: about "
                f"{format_impressions(total_impressions)} impressions against the "
                f"{format_impressions(requirements.impression_goal)} impression goal."
            )
        else:
            tradeoffs.append(
                f"Forecasted delivery: about {format_impressions(total_impressions)} impressions."
            )

        if selected is not None:
            tradeoffs.append(
                f"Best single-product option: {selected.product.product_name} at "
                f"{format_percent(selected.score(requirements.primary_kpi))}."
            )

        return tuple(tradeoffs)

    @staticmethod
    def _single_product_rationale(
        selected: ProductCandidate,
        requirements: ClientRequirements,
    ) -> str:
        assert requirements.primary_kpi is not None
        assert requirements.budget is not None

        kpi_name = KPI_LABELS[requirements.primary_kpi]
        kpi_short_name = KPI_SHORT_LABELS[requirements.primary_kpi]
        kpi_value = selected.score(requirements.primary_kpi)

        return (
            f"Recommend {selected.product.product_name} for {kpi_name}. It has the strongest "
            f"viable {kpi_short_name} benchmark at {format_percent(kpi_value)} and can use the "
            f"full {format_currency(requirements.budget)} budget."
        )

    @staticmethod
    def _single_product_budget_caveat_rationale(
        selected: ProductCandidate,
        recommended: RecommendedProduct,
        requirements: ClientRequirements,
    ) -> str:
        assert requirements.primary_kpi is not None
        assert requirements.budget is not None

        kpi_name = KPI_LABELS[requirements.primary_kpi]
        kpi_short_name = KPI_SHORT_LABELS[requirements.primary_kpi]
        kpi_value = selected.score(requirements.primary_kpi)
        caveats = []

        if recommended.budget < requirements.budget:
            remaining_budget = requirements.budget - recommended.budget
            caveats.append(
                f"It can use about {format_currency(recommended.budget)} of the "
                f"{format_currency(requirements.budget)} budget; a single-product plan would "
                f"leave about {format_currency(remaining_budget)} unallocated."
            )

        if (
            requirements.impression_goal is not None
            and recommended.forecasted_impressions < requirements.impression_goal
        ):
            caveats.append(
                f"It forecasts about {format_impressions(recommended.forecasted_impressions)} "
                f"impressions, below the {format_impressions(requirements.impression_goal)} "
                "impression goal."
            )

        if not caveats:
            caveats.append("It has planning constraints, but remains the strongest single-product fit.")

        return (
            f"Recommend {selected.product.product_name}. It is the strongest single-product "
            f"fit for {kpi_name} at {format_percent(kpi_value)}. "
            + " ".join(caveats)
        )

    @staticmethod
    def _bundle_rationale(
        bundle: list[RecommendedProduct],
        requirements: ClientRequirements,
    ) -> str:
        assert requirements.primary_kpi is not None
        assert requirements.budget is not None

        product_names = " + ".join(product.product_name for product in bundle)
        blended_kpi = RecommendationEngine._plan_kpi(bundle, requirements.primary_kpi)
        kpi_name = KPI_LABELS[requirements.primary_kpi]
        return (
            f"Recommend {product_names}. This bundle has the strongest blended {kpi_name} "
            f"at {format_percent(blended_kpi)} and uses the full "
            f"{format_currency(requirements.budget)} budget."
        )
