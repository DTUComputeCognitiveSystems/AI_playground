class TextModifier:
    def __init__(self, start, end, field_name, field_value):
        self.field_name = field_name
        self.field_value = field_value
        self.start = start
        self.end = end

    def offset(self, start_offset, end_offset=None):
        if end_offset is None:
            end_offset = start_offset
        return TextModifier(
            start=self.start + start_offset,
            end=self.end + end_offset,
            field_name=self.field_name,
            field_value=self.field_value,
        )

    def __str__(self):
        return "{}({}, {}, {})".format(type(self).__name__, self.start, self.end, self.field_value)

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        return self.start < other.start

    def __le__(self, other):
        return self.start <= other.start
