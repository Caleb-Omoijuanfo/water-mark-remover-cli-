"""Region parsing and validation for watermark bounding boxes."""

from __future__ import annotations

from dataclasses import dataclass

from videoclean.exceptions import InvalidRegionError


@dataclass(frozen=True, slots=True)
class Region:
    """Axis-aligned rectangle in pixel coordinates.

    Coordinates follow the convention ``x1,y1,x2,y2`` where ``(x1, y1)`` is the
    inclusive top-left corner and ``(x2, y2)`` is the exclusive bottom-right
    corner (OpenCV-style half-open intervals for width/height).
    """

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        self._validate_geometry()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def parse(cls, value: str) -> Region:
        """Parse a region from an ``x1,y1,x2,y2`` string.

        Raises
        ------
        InvalidRegionError
            If the string is malformed or the resulting geometry is invalid.
        """
        if value is None or not str(value).strip():
            raise InvalidRegionError(
                "Region is empty. Expected format: x1,y1,x2,y2 "
                "(e.g. 20,20,300,120)."
            )

        raw = str(value).strip()
        parts = [part.strip() for part in raw.split(",")]

        if len(parts) != 4:
            raise InvalidRegionError(
                f"Region must have exactly 4 comma-separated integers "
                f"(x1,y1,x2,y2), got {len(parts)} value(s): {value!r}."
            )

        coords: list[int] = []
        for index, part in enumerate(parts):
            label = ("x1", "y1", "x2", "y2")[index]
            if part == "" or any(ch in part for ch in " \t"):
                # empty already handled by strip; keep explicit message
                raise InvalidRegionError(
                    f"Region component {label} is empty or malformed in {value!r}."
                )
            try:
                # Reject floats like "1.5" — regions are integer pixels.
                if any(ch in part for ch in ".eE"):
                    raise ValueError("non-integer")
                number = int(part, 10)
            except ValueError as exc:
                raise InvalidRegionError(
                    f"Region component {label} must be an integer, "
                    f"got {part!r} in {value!r}."
                ) from exc
            coords.append(number)

        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        """Width in pixels (x2 - x1)."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Height in pixels (y2 - y1)."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Area in pixels."""
        return self.width * self.height

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_geometry(self) -> None:
        for name, value in (
            ("x1", self.x1),
            ("y1", self.y1),
            ("x2", self.x2),
            ("y2", self.y2),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise InvalidRegionError(
                    f"Region component {name} must be an integer, got {value!r}."
                )
            if value < 0:
                raise InvalidRegionError(
                    f"Region component {name} must be >= 0, got {value}."
                )

        if self.x1 >= self.x2:
            raise InvalidRegionError(
                f"Region requires x1 < x2, got x1={self.x1}, x2={self.x2}."
            )
        if self.y1 >= self.y2:
            raise InvalidRegionError(
                f"Region requires y1 < y2, got y1={self.y1}, y2={self.y2}."
            )

    def validate_bounds(self, frame_width: int, frame_height: int) -> None:
        """Ensure this region lies fully inside a frame of the given size.

        Parameters
        ----------
        frame_width:
            Frame width in pixels (must be > 0).
        frame_height:
            Frame height in pixels (must be > 0).

        Raises
        ------
        InvalidRegionError
            If the region extends outside the frame or dimensions are invalid.
        """
        if frame_width <= 0 or frame_height <= 0:
            raise InvalidRegionError(
                f"Frame dimensions must be positive, "
                f"got {frame_width}x{frame_height}."
            )

        if self.x2 > frame_width or self.y2 > frame_height:
            raise InvalidRegionError(
                f"Region ({self.x1},{self.y1},{self.x2},{self.y2}) is outside "
                f"the frame bounds ({frame_width}x{frame_height}). "
                f"Ensure x2 <= {frame_width} and y2 <= {frame_height}."
            )

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return ``(x1, y1, x2, y2)``."""
        return (self.x1, self.y1, self.x2, self.y2)

    def __str__(self) -> str:
        return f"{self.x1},{self.y1},{self.x2},{self.y2}"
