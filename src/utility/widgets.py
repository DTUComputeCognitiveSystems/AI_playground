import ipywidgets as widgets
from IPython.display import display


def TextAreaSubmit(submit_func=print, default_text=''):
    ''' displays a text area and a submit button for
        evalutating 'submit_func' on the given text
    '''

    text_layout = widgets.Layout(height='150px')

    text_area = widgets.Textarea(value=default_text, layout=text_layout)

    submit_button = widgets.Button(
        description='Submit',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Submit'
    )

    box_layout = widgets.Layout(display='flex',
                                flex_flow='column',
                                align_items='stretch',
                                width='100%')

    box = widgets.Box(children=[text_area, submit_button], layout=box_layout)

    display(box)

    def _submit_function(b):
        submit_func(text_area.value)

    submit_button.on_click(_submit_function)


def SliderParameters(submit_func=print, *args):

    slider_widgets = []

    for arg in args:
        slider_widgets.append(widgets.IntSlider(
            value=arg['min'],
            min=arg['min'],
            max=arg['max'],
            step=1,
            description=arg['name'],
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        ))

    submit_button = widgets.Button(
        description='Submit',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Submit'
    )

    box_layout = widgets.Layout(display='flex',
                                flex_flow='column',
                                align_items='stretch',
                                width='100%')

    box = widgets.Box(children=slider_widgets +
                      [submit_button], layout=box_layout)

    display(box)

    def _submit_function(b):
        param_vals = {}
        for slider_widget in slider_widgets:
            param_vals[slider_widget.description] = slider_widget.value
        submit_func(param_vals)

    submit_button.on_click(_submit_function)
