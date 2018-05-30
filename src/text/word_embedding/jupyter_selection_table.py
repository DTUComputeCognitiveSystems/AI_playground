import pandas as pd
from ipywidgets import Button, HBox, VBox, Label
from ipywidgets import HTML


class SelectionTable:
    def __init__(self, data, row_height=28, column_width=80, row_button_width=50, table_cell_paddings=8.16,
                 button_padding="0pt 0pt 0pt 0pt"):
        """
        Class for creating a pandas table in jupyter in which specific rows and columns can be selected.
        :param pd.DataFrame data: The data for the table. Made for text-elements but may also work for other types.
        :param int row_height: Height of rows in pt.
        :param int column_width: Width of columns in pt.
        :param int row_button_width: Width of buttons in rows in pt.
        :param int table_cell_paddings: Padding of cells in table in pt.
        :param str button_padding: CSS padding of buttons.
        """
        # Store dimensions and convert to strings for html
        self._row_height = row_height
        self._column_width = column_width
        self._row_button_width = row_button_width
        self._table_cell_paddings = table_cell_paddings
        self.row_height = "{}pt".format(self._row_height)
        self.column_width = "{}pt".format(self._column_width)
        self.row_button_width = "{}pt".format(self._row_button_width)
        self.table_cell_paddings = "{}pt".format(self._table_cell_paddings)
        self.button_padding = button_padding

        # Initialize
        self.observers = []
        self.df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

        # Make a button for each row
        self.row_button_list = [
            Button(
                description='select',
                layout=dict(height=self.row_height,
                            width=self.row_button_width,
                            padding=self.button_padding),
            ) for _ in range(self.df.shape[0])
        ]
        self.row_on = [True] * self.df.shape[0]
        for nr, button in enumerate(self.row_button_list):
            button.on_click(self._row_button_clicked(nr=nr))

        # Make a button for each column
        self.column_button_list = [
            Button(
                description='select',
                layout=dict(height=self.row_height,
                            width=self.column_width,
                            padding=self.button_padding),
            ) for _ in range(self.df.shape[1])
        ]
        self.col_on = [True] * self.df.shape[1]
        for nr, button in enumerate(self.column_button_list):
            button.on_click(self._col_button_clicked(nr=nr))

        # Compute total width of table
        self.table_width = max((self.df.shape[1] + 1) * self._column_width + self._row_button_width, 500)

        # Make the table
        self._make_table()

    def observe(self, method):
        """
        Pass a method for being called when table is interacted with.
        :param Callable method:
        :return:
        """
        self.observers.append(method)

    def _note_change(self):
        """
        Notify observers.
        """
        for method in self.observers:
            method()

    def _row_button_clicked(self, nr):
        """
        Note click of a button in a row.
        :param nr:
        """
        def set_bool(_=None):
            self.row_on[nr] = not self.row_on[nr]
            self._make_table()
            self._note_change()

        return set_bool

    def _col_button_clicked(self, nr):
        """
        Note click of a button in a column.
        :param nr:
        """
        def set_bool(_=None):
            self.col_on[nr] = not self.col_on[nr]
            self._make_table()
            self._note_change()

        return set_bool

    def selection_colorize(self, _=None):
        """
        Colorize table depending on selection on rows and columns.
        :param _:
        """
        formatters = pd.DataFrame([
            ["color: black" if row and col else "color: lightgrey" for col in self.col_on]
            for row in self.row_on
        ])
        return formatters

    def _make_table(self):
        # Global styles
        styles = [
            dict(selector="tr", props=[
                ("height", self.row_height),
                ("vertical-align", "middle"),
                ("text-align", "center"),
                ("width", self.column_width),
            ]),
            dict(selector="td", props=[
                ("padding-bottom", self.table_cell_paddings),
                ("padding-top", self.table_cell_paddings),
                ("vertical-align", "middle"),
                ("text-align", "center"),
                ("width", self.column_width),
            ]),
            dict(selector="th", props=[
                ("padding-bottom", self.table_cell_paddings),
                ("padding-top", self.table_cell_paddings),
                ("vertical-align", "middle"),
                ("text-align", "center"),
                ("width", self.column_width),
            ]),
        ]

        # Make styler for DataFrame
        styler = self.df.style.set_table_attributes('class="table"') \
            .set_table_styles(styles) \
            .apply(self.selection_colorize, axis=None)

        # Render HTML table from DataFrame
        html = styler.render()

        # Build visual components
        v_row_buttons = VBox(
            [Label(description='', layout=dict(height=self.row_height,
                                               width=self.row_button_width,
                                               padding=self.button_padding)),
             *self.row_button_list],
        )
        v_bottom = HBox([
            v_row_buttons,
            HTML(html)
        ])
        v_col_buttons = HBox(
            [Label(description='', layout=dict(height=self.row_height,
                                               width=self.row_button_width,
                                               padding=self.button_padding)),
             Label(description='', layout=dict(height=self.row_height,
                                               width=self.column_width,
                                               padding=self.button_padding)),
             *self.column_button_list],
        )

        # Make dashboard
        self._table = VBox(
            (v_col_buttons, v_bottom,),
            layout=dict(width="{}pt".format(self.table_width))
        )

    @property
    def table(self):
        return self._table