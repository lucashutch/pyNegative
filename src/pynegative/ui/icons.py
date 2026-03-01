from functools import lru_cache
from pathlib import Path

from PySide6 import QtCore, QtGui, QtSvg

_ICON_DIR = Path(__file__).resolve().parent / "assets" / "icons"
_DEFAULT_COLOR = "#d4d4d8"


def _normalize_color(color: str | None) -> str:
    if not color:
        return _DEFAULT_COLOR
    qcolor = QtGui.QColor(color)
    if not qcolor.isValid():
        return _DEFAULT_COLOR
    name_format = (
        QtGui.QColor.NameFormat.HexArgb
        if qcolor.alpha() < 255
        else QtGui.QColor.NameFormat.HexRgb
    )
    return qcolor.name(name_format)


def _icon_path(name: str) -> Path:
    return _ICON_DIR / "24" / "outline" / f"{name}.svg"


@lru_cache(maxsize=512)
def _render_pixmap(name: str, size: int, color: str) -> QtGui.QPixmap:
    icon_path = _icon_path(name)
    if not icon_path.exists() or size <= 0:
        return QtGui.QPixmap()

    svg = icon_path.read_text(encoding="utf-8")
    svg = svg.replace("currentColor", color)
    svg = svg.replace("#0F172A", color)
    svg = svg.replace("#111827", color)
    svg = svg.replace("#0f172a", color)

    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode("utf-8")))
    if not renderer.isValid():
        return QtGui.QPixmap()

    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)

    painter = QtGui.QPainter(pixmap)
    renderer.render(painter)
    painter.end()

    return pixmap


def get_heroicon(name: str, size: int = 20, color: str | None = None) -> QtGui.QIcon:
    normalized_color = _normalize_color(color)
    pixmap = _render_pixmap(name, int(size), normalized_color)
    if pixmap.isNull():
        return QtGui.QIcon()
    return QtGui.QIcon(pixmap)
