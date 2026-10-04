"""Attacker scenarios describing externally available information."""

from __future__ import annotations

from dataclasses import dataclass, field

from maskhub.errors import InvalidRequest


def _check_probability(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("availability must be a number")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError("availability must be within [0, 1]")


@dataclass(frozen=True)
class Scenario:
    """External-information model: how available each quasi-identifier is."""

    name: str
    availability: float
    field_availability: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("scenario name is required")
        _check_probability(self.availability)
        for key, value in self.field_availability.items():
            if not key.strip():
                raise ValueError("field_availability keys must be non-empty")
            _check_probability(value)

    def field_probability(self, field_name: str) -> float:
        return float(self.field_availability.get(field_name, self.availability))

    def coverage(self, fields: tuple[str, ...]) -> float:
        """Probability that external sources supply the whole QI combination."""

        probability = 1.0
        for name in fields:
            probability *= self.field_probability(name)
        return probability

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "availability": self.availability,
            "field_availability": dict(self.field_availability),
        }


BUILTIN_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(name="targeted", availability=1.0),
    Scenario(name="ambient", availability=0.5),
)


def parse_scenarios(raw: object) -> tuple[Scenario, ...]:
    if raw is None:
        return BUILTIN_SCENARIOS
    if not isinstance(raw, list) or not raw:
        raise InvalidRequest("scenarios must be a non-empty list")
    scenarios: list[Scenario] = []
    for item in raw:
        if not isinstance(item, dict):
            raise InvalidRequest("each scenario must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise InvalidRequest("scenario name is required")
        overrides = item.get("field_availability", {})
        if not isinstance(overrides, dict):
            raise InvalidRequest("field_availability must be an object")
        try:
            scenarios.append(
                Scenario(
                    name=name.strip(),
                    availability=item.get("availability", 0.5),
                    field_availability={str(key): value for key, value in overrides.items()},
                )
            )
        except ValueError as exc:
            raise InvalidRequest(str(exc)) from exc
    return tuple(scenarios)
