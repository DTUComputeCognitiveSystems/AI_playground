from pathlib import Path

from src.text.utility.text_modifiers import TextModifier


class HTMLTag:
    def __init__(self, contents=None):
        if isinstance(contents, (str, HTMLTag)):
            self.contents = [contents]  # type: list
        else:
            self.contents = list(contents)

    def append(self, contents):
        self.contents.append(contents)

    @property
    def formatter(self) -> str:
        raise NotImplementedError

    def __str__(self):
        return self.formatter.format("".join([str(val) for val in self.contents])).replace("\n", "<br>\n")


class _Doc(HTMLTag):
    @property
    def formatter(self):
        return "<html>\n<body>\n{}\n</body>\n</html>"


class _Ital(HTMLTag):
    @property
    def formatter(self):
        return "<i>{}</i>"


class _Bold(HTMLTag):
    @property
    def formatter(self):
        return "<b>{}</b>"


class _FontSize(HTMLTag):
    def __init__(self, contents, size):
        super().__init__(contents=contents)
        self.size = size

    @property
    def formatter(self):
        return "<font size={}>{{}}</font>".format(self.size)


class CharSet(HTMLTag):
    def __init__(self, contents, character_set="UTF-8"):
        super().__init__(contents=contents)
        self.character_set = character_set

    @property
    def formatter(self):
        return "<form accept-charset=\"{}\">{{}}</form>".format(self.character_set)


class Pre(HTMLTag):
    @property
    def formatter(self):
        return "<pre>{}</pre>"


class _Color(HTMLTag):
    def __init__(self, contents, color):
        super().__init__(contents=contents)

        if isinstance(color, str):
            self.color = color
        else:
            self.color = "#{0:02x}{1:02x}{2:02x}".format(*[self.clamp(val) for val in color])

    @staticmethod
    def clamp(x):
        if isinstance(x, float):
            x = x * 255
        return max(0, min(int(x), 255))

    @property
    def formatter(self):
        return '<font color="{}">{{}}</font>'.format(self.color)


def modified_text2html(text: str, modifiers: dict):
    if modifiers.get("weight", None) == "bold":
        text = _Bold(text)
    if modifiers.get("style", None) == "italic":
        text = _Ital(text)
    if "color" in modifiers:
        text = _Color(text, modifiers["color"])
    if "html_fontsize" in modifiers:
        text = _FontSize(text, modifiers["html_fontsize"])
    return text


def modified_text_to_html(text, modifiers, html_fontsize=None, string=True):
    # Determine all splits for modifiers
    splits = list(sorted(set([
        max(val, 0)
        for modifier in modifiers
        for val in (modifier.start, modifier.end)

    ])))

    # Compute section modifications
    modifier_locs = {val: idx for idx, val in enumerate(splits)}
    section_mods = [dict() for _ in range(len(splits) + 1)]
    for modifier in modifiers:
        # Only consider positive numbers
        start = max(modifier.start, 0)
        end = max(modifier.end, 0)

        # Note modifications for sections.
        for section in section_mods[modifier_locs[start] + 1:modifier_locs[end] + 1]:
            section[modifier.field_name] = modifier.field_value

    # Split text into sections
    sections = [text[start:end] for start, end in zip([0] + splits, splits + [None])]

    # Go through sections
    html_parts = []
    for section, section_mod in zip(sections, section_mods):
        if html_fontsize is not None:
            section_mod["html_fontsize"] = html_fontsize

        html_parts.append(modified_text2html(section, section_mod))

    doc = _Doc(html_parts)

    if string:
        return str(doc)
    return doc


if __name__ == "__main__":

    manual_lines = False

    with Path("src", "text", "font_info", "test_text.txt").open("r") as file:
        text_lines = file.readlines()
        text_lines = text_lines[:3]
        the_text = "".join(text_lines)

    if manual_lines:
        text_lines = [
            "",
            "\nhey there",
            "how are you?",
            "why are you asking me?!?",
            "dude I was just being polite, take a chill pill",
        ]

        the_text = "\n".join(text_lines)

    letter_modifiers = dict(
        e=dict(weight="bold", color=(1., 0., 0.), style="italic"),
        a=dict(weight="bold", color=(0., 0., 1.)),
        t=dict(color=(1., 0., 1.)),
        s=dict(color=(0., .7, 0.), style="italic"),
    )

    the_modifiers = []
    for letter, formatting in letter_modifiers.items():
        the_modifiers.extend([
            TextModifier(val, val + 1, item, value)
            for val, char in enumerate(the_text)
            for item, value in formatting.items()
            if char == letter
        ])

    html = modified_text_to_html(
        text=the_text,
        modifiers=the_modifiers
    )

    with Path("src", "text", "font_info", "test_html.html").open("w") as file:
        file.write(html)
