from .compat import create_star_pixmap
from .starrating import StarRatingWidget


class PreviewStarRatingWidget(StarRatingWidget):
    """A larger star rating widget for preview mode with 30px stars."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)

    def _create_star_pixmap(self, filled):
        return create_star_pixmap(filled, size=30, font_size=24)
