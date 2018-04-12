import matplotlib.pyplot as plt
import matplotlib.text

from src.image.video.base import VideoFlair, _Video, SimpleVideo


class VideoTexter(VideoFlair):
    def __init__(self, initial_text="", backgroundcolor="darkblue", color="white", position="sw"):
        super().__init__([])
        position = position.lower().strip()
        self._initial_text = initial_text

        # Vertical
        if position in ("sw", "s", "se"):
            self._verticalalignment = 'bottom'
            self._y = 0.0
        elif position in ("w", "e"):
            self._verticalalignment = 'center'
            self._y = 0.5
        else:
            self._verticalalignment = 'top'
            self._y = 1.0

        # Horizontal
        if position in ("sw", "w", "nw"):
            self._horizontalalignment = 'left'
            self._x = 0.0
        elif position in ("n", "s"):
            self._horizontalalignment = 'center'
            self._x = 0.5
        else:
            self._horizontalalignment = 'right'
            self._x = 1.0

        self._backgroundcolor = backgroundcolor
        self._color = color
        self._text = None  # type: matplotlib.text.Text

    def initialize(self):
        self._text = plt.text(
            x=self._x,
            y=self._y,
            s=self._initial_text,
            horizontalalignment=self._horizontalalignment,
            verticalalignment=self._verticalalignment,
            transform=plt.gca().transAxes,
            backgroundcolor=self._backgroundcolor,
            color=self._color
        )
        self._artists.append(self.text)

    def update(self, video):
        """
        :param _Video video:
        :return:
        """
        self.text.set_text("Text")

    def set_text(self, s):
        self.text.set_text(s)

    def set_background_color(self, new_color):
        self._text.set_backgroundcolor(new_color)

    def color(self, new_color):
        self._text.set_color(new_color)

    @property
    def text(self):
        return self._text


if __name__ == "__main__":

    video = SimpleVideo(
        video_length=10
    )
    video.add_flair(VideoTexter())
    video.start()
