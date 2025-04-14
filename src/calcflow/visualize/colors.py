import colorsys
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class Color:
    """Represents a color in hexadecimal format."""

    hex_code: str

    def __post_init__(self) -> None:
        """Validate the hex code format."""
        if not isinstance(self.hex_code, str):
            raise TypeError("Hex code must be a string")

        # Remove # if present
        hex_value = self.hex_code.lstrip("#")

        # Check length and valid hex characters
        if len(hex_value) != 6 or not all(c in "0123456789ABCDEFabcdef" for c in hex_value):
            raise ValueError(f"Invalid hex color code: {self.hex_code}")

    @property
    def rgb(self) -> tuple[int, int, int]:
        """Convert hex to RGB tuple."""
        hex_value = self.hex_code.lstrip("#")
        r, g, b = (int(hex_value[i : i + 2], 16) for i in (0, 2, 4))
        return r, g, b

    @property
    def hsv(self) -> tuple[float, float, float]:
        """Convert to HSV for hue-based comparisons."""
        r, g, b = self.rgb
        return colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)

    def __str__(self) -> str:
        return self.hex_code


@dataclass
class ColorPalette:
    """A collection of colors forming a palette."""

    colors: list[Color]

    def __post_init__(self) -> None:
        """Validate the colors list."""
        if not isinstance(self.colors, list):
            raise TypeError("Colors must be a list")
        if not all(isinstance(color, Color) for color in self.colors):
            raise TypeError("All elements must be Color instances")

    @classmethod
    def from_hex_codes(cls, hex_codes: list[str]) -> "ColorPalette":
        """
        Create a ColorPalette from a list of hex color codes.

        Args:
            hex_codes: List of hex color strings (with or without # prefix)

        Returns:
            A new ColorPalette instance
        """
        return cls([Color(hex_code) for hex_code in hex_codes])

    def __len__(self) -> int:
        return len(self.colors)

    def iter_n(self, count: int, reverse: bool = False) -> Iterator[Color]:
        """
        Generate a sequence of colors distributed evenly across the palette.

        Args:
            count: Number of colors to select
            reverse: Whether to start from the end of the palette

        Returns:
            Iterator yielding selected colors
        """
        n = len(self.colors)
        colors = list(self.colors)

        # If reversed, reverse the color list
        if reverse:
            colors.reverse()

        # If requesting more colors than available, cycle through all colors
        if count >= n:
            for i in range(count):
                yield colors[i % n]
        # Otherwise distribute selections evenly
        else:
            # Calculate the stride to maximize distance between selected colors
            stride = n / count
            for i in range(count):
                # Use floating-point math for the index calculation, then round to nearest integer
                index = round(i * stride)
                if index >= n:  # Safeguard against any rounding issues
                    index = n - 1
                yield colors[index]

    def sample_n(self, count: int, reverse: bool = False) -> list[Color]:
        """
        Select distributed colors from the palette.

        Args:
            count: Number of colors to select
            reverse: Whether to start from the end of the palette

        Returns:
            List of selected Color objects
        """
        return list(self.iter_n(count, reverse=reverse))

    def sample_n_hex(self, count: int, reverse: bool = False) -> list[str]:
        """
        Select distributed colors from the palette and return as hex codes.

        Args:
            count: Number of colors to select
            reverse: Whether to start from the end of the palette

        Returns:
            List of selected hex color codes
        """
        selected = self.sample_n(count, reverse=reverse)
        return [color.hex_code for color in selected]


blue_4 = ["#0000ff", "#4040ff", "#8080ff", "#bfbfff"]
blue_4_dark = ["#00002e", "#000062", "#000096", "#0000cb"]
purple_4 = ["#7209b7", "#8e2cc5", "#aa4fd4", "#c771e2"]
purple_4_dark = ["#0e0016", "#27023e", "#400466", "#59078f"]
red_4 = ["#d80032", "#df234e", "#e6476a", "#ed6a85"]
red_4_dark = ["#560000", "#76000c", "#970019", "#b80026"]
green_4 = ["#006400", "#337b02", "#669204", "#99a806"]
green_4_dark = ["#002400", "#003400", "#004400", "#005400"]
# fmt:off
blue_20 = ColorPalette.from_hex_codes(
    ["#0026D0","#002FD2","#0039D5","#0042D7","#004CDA","#0055DC","#005FDF","#0068E1","#0071E4","#007BE6",
     "#0084E9","#008EEB","#0097EE","#00A0F0","#00AAF3","#00B3F5","#00BDF8","#00C6FA","#00D0FD","#00D9FF"])
red_20 = ColorPalette.from_hex_codes(
    ["#590000","#620808","#6A0F0F","#731717","#7C1E1E","#852626","#8D2D2D","#963535","#9F3D3D","#A84444",
     "#B04C4C","#B95353","#C25B5B","#CB6363","#D36A6A","#DC7272","#E57979","#EE8181","#F68888","#FF9090"])
purple_20 = ColorPalette.from_hex_codes(
    ["#430059","#4D0862","#560F6A","#601773","#691E7C","#732685","#7C2D8D","#863596","#8F3C9F","#9944A8",
     "#A24BB0","#AC53B9","#B55AC2","#BF62CB","#C869D3","#D271DC","#DB78E5","#E580EE","#EE87F6","#F88FFF"])
coral_20 = ColorPalette.from_hex_codes(
    ["#590021","#620827","#6A0F2D","#731732","#7C1E38","#85263E","#8D2D44","#96354A","#9F3C4F","#A84455",
     "#B04B5B","#B95361","#C25A66","#CB626C","#D36972","#DC7178","#E5787E","#EE8083","#F68789","#FF8F8F"])
green_20 = ColorPalette.from_hex_codes(
    ["#035900","#0C6208","#156A0F","#1E7317","#267C1E","#2F8526","#388D2D","#419635","#4A9F3C","#53A844",
     "#5BB04B","#64B953","#6DC25A","#76CB62","#7FD369","#88DC71","#90E578","#99EE80","#A2F687","#ABFF8F"])
orange_20 = ColorPalette.from_hex_codes(
    ["#C76A00","#CA7008","#CD7710","#D07D18","#D38320","#D68927","#D9902F","#DC9637","#DF9C3F","#E2A247",
     "#E4A94F","#E7AF57","#EAB55F","#EDBB67","#F0C26F","#F3C876","#F6CE7E","#F9D486","#FCDB8E","#FFE196"])

green_yellow_20 = ColorPalette.from_hex_codes(
    ['#006400', '#0d6a00', '#1a6f01', '#267502', '#337b02', '#408002', '#4c8603', '#598c03', '#669204', '#739704', 
     '#809d05', '#8ca306', '#99a806', '#a6ae06', '#b2b407', '#bfba08', '#ccbf08', '#d9c508', '#e6cb09', '#f2d00a'])

blue_red_20 = ColorPalette.from_hex_codes([
    '#2f00ff', '#3906f8', '#440cf1', '#4e11ea', '#5917e3', '#631ddc', '#6d22d5', '#7828ce', '#822ec7', '#8d34c0', 
    '#973ab9', '#a13fb2', '#ac45ab', '#b64ba4', '#c1509d', '#cb5696', '#d55c8f', '#e06288', '#ea6881', '#f56d7a'])
# fmt:on
