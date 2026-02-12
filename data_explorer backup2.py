import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import plotly.graph_objects as go

import webbrowser
import pandas as pd
from os import listdir, getcwd, mkdir, makedirs
from os.path import join, isdir, exists
from shutil import copy2
from update_conf import update_csv_file


def get_projects_list():
    global projects_list
    projects_folder = join(getcwd(), 'main', 'assets')
    projects_list = [f for f in listdir(projects_folder) if isdir(join(projects_folder, f))]
    projects_list.sort()


def update_classes_project(batches_list, project_name):
    global classes_per_batch

    temp_classes_project = {}
    temp_classes_project_total = {}  # Store total counts (unfiltered)

    print("Update classes project:")
    print(f"Batches: {batches_list}")
    print(f"Classes per batch keys: {list(classes_per_batch.keys())}")

    for i, file in enumerate(batches_list):
        if file in classes_per_batch:
            dict_temp = classes_per_batch[file]
            print(f"File {file}: {dict_temp}")
            for keys in dict_temp.keys():
                if keys not in temp_classes_project.keys():
                    temp_classes_project[keys] = dict_temp[keys]
                else:
                    temp_classes_project[keys] += dict_temp[keys]
        else:
            print(f"Warning: {file} not found in classes_per_batch")

    # Calculate total counts (unfiltered) for each class
    batches_folder = join(getcwd(), 'main', 'assets', project_name, 'dataframes')
    for file in batches_list:
        file_path = join(batches_folder, file)
        df_temp = pd.read_csv(file_path)
        dict_temp = dict(df_temp['manual_label'].value_counts())

        for keys in dict_temp.keys():
            if keys not in temp_classes_project_total.keys():
                temp_classes_project_total[keys] = dict_temp[keys]
            else:
                temp_classes_project_total[keys] += dict_temp[keys]

    temp_keys = list(temp_classes_project.keys())
    temp_keys.sort()
    counts = [temp_classes_project[k] for k in temp_keys]
    total_counts = [temp_classes_project_total.get(k, 0) for k in temp_keys]
    classes_text = [{'label': f' {k} ({c}/{t})', 'value': 'enable'} for k, c, t in zip(temp_keys, counts, total_counts)]
    classes_list = [k for k in temp_keys]

    print(f"Final counts: {dict(zip(temp_keys, counts))}")
    print(f"Total counts: {dict(zip(temp_keys, total_counts))}")
    print()

    return classes_text, classes_list


def get_classes(batches_folder, batches_list):
    classes_list = []
    for file in batches_list:
        file_path = join(batches_folder, file)
        df_temp = pd.read_csv(file_path)
        classes_list.extend(df_temp['manual_label'].unique())
    classes_list = list(set(classes_list))
    classes_list.sort()
    return classes_list


def get_histogram_data_metric(batches_folder, batches_list, classes_list, metric_col):
    """Generate histogram data for each class based on a metric distribution (e.g. manual_conf or var_norm)."""
    global loadbar_classes
    hist_data = {label: [] for label in classes_list}

    for i, file in enumerate(batches_list):
        file_path = join(batches_folder, file)
        df_temp = pd.read_csv(file_path)

        # Ensure metric exists
        if metric_col not in df_temp.columns:
            if metric_col == 'manual_conf':
                # try to fix missing manual_conf using your updater
                try:
                    print(f"'{metric_col}' missing in {file}, attempting update_csv_file...")
                    df_temp = update_csv_file(file_path)
                except Exception as e:
                    print(f"Could not update manual_conf for {file}: {e}")
                    df_temp[metric_col] = 0.0
            else:
                # for var_norm, we can't derive it here; fill with 0.0
                df_temp[metric_col] = 0.0

        # coerce numeric
        df_temp[metric_col] = pd.to_numeric(df_temp[metric_col], errors='coerce').fillna(0.0)

        for label in classes_list:
            vals = df_temp[df_temp['manual_label'] == label][metric_col].tolist()
            hist_data[label].extend(vals)

        loadbar_classes = 20 + 60 * i / max(1, len(batches_list))

    return hist_data


def get_classes_per_batch(batches_folder, batches_list, classes_list, slider_values_conf=None, slider_values_var=None, changed_slider_index=None):
    global classes_per_batch, loadbar_classes

    print()
    print("Get classes per batch")
    print()

    if changed_slider_index is None:
        classes_per_batch = {}

    # defaults
    default_conf = [0.85, 1.0]
    default_var = [0.0, 1.0]

    for file in batches_list:
        file_path = join(batches_folder, file)
        df_temp = pd.read_csv(file_path)

        # Ensure required columns exist
        if 'manual_conf' not in df_temp.columns:
            try:
                print(f"'manual_conf' missing in {file}, updating...")
                df_temp = update_csv_file(file_path)
            except Exception as e:
                print(f"Error updating manual_conf in {file}: {e}")
                df_temp['manual_conf'] = 0.0

        if 'var_norm' not in df_temp.columns:
            # can't infer here; fill 0.0 so it won't pass high thresholds
            df_temp['var_norm'] = 0.0

        df_temp['manual_conf'] = pd.to_numeric(df_temp['manual_conf'], errors='coerce').fillna(0.0)
        df_temp['var_norm'] = pd.to_numeric(df_temp['var_norm'], errors='coerce').fillna(0.0)

        if file not in classes_per_batch:
            classes_per_batch[file] = {}

        for i, label in enumerate(classes_list):
            if changed_slider_index is not None and i != changed_slider_index and (label in classes_per_batch[file]):
                continue

            conf_range = default_conf
            if slider_values_conf is not None and len(slider_values_conf) == len(classes_list) and slider_values_conf[i] is not None:
                conf_range = slider_values_conf[i]

            var_range = default_var
            if slider_values_var is not None and len(slider_values_var) == len(classes_list) and slider_values_var[i] is not None:
                var_range = slider_values_var[i]

            filtered = df_temp[
                (df_temp['manual_label'] == label) &
                (df_temp['manual_conf'] >= conf_range[0]) &
                (df_temp['manual_conf'] < conf_range[1]) &
                (df_temp['var_norm'] >= var_range[0]) &
                (df_temp['var_norm'] < var_range[1])
            ]
            classes_per_batch[file][label] = len(filtered)

        loadbar_classes = 20

    loadbar_classes = 100


def _make_range_folder_suffix(metric_name, rng):
    """Helper for naming folders based on slider ranges."""
    if rng is None:
        return ""
    a, b = rng
    a_i = int(round(a * 100))
    b_i = int(round(b * 100))
    if b >= 1.0:
        return f"{metric_name}>={a_i}"
    return f"{metric_name}>={a_i}and<{b_i}"


def save_dataset(output_folder, project_name, selected_classes, selected_batches, slider_values_conf=None, slider_values_var=None):
    global loadbar_save_dataset

    print("Save dataset")
    print(f"Selected classes: {selected_classes}")
    print(f"Selected batches: {selected_batches}")
    print(f"Slider values conf: {slider_values_conf}")
    print(f"Slider values var: {slider_values_var}")
    print()

    count = 0
    project_folder = join(getcwd(), 'main', 'assets', project_name)
    batches_folder = join(project_folder, 'dataframes')
    images_folder = join(project_folder, 'images')

    if not exists(output_folder):
        mkdir(output_folder)

    # defaults
    default_conf = [0.85, 1.0]
    default_var = [0.0, 1.0]

    # Create folders for each class (filtered + remaining) with both criteria in the name
    for label in selected_classes:
        class_index = selected_classes.index(label)

        conf_rng = default_conf
        if slider_values_conf is not None and class_index < len(slider_values_conf) and slider_values_conf[class_index] is not None:
            conf_rng = slider_values_conf[class_index]

        var_rng = default_var
        if slider_values_var is not None and class_index < len(slider_values_var) and slider_values_var[class_index] is not None:
            var_rng = slider_values_var[class_index]

        conf_suffix = _make_range_folder_suffix("conf", conf_rng)
        var_suffix = _make_range_folder_suffix("var", var_rng)

        filtered_folder_name = f"{label}_{conf_suffix}_{var_suffix}".replace("__", "_").strip("_")
        filtered_label_path = join(output_folder, filtered_folder_name)
        if not exists(filtered_label_path):
            mkdir(filtered_label_path)

        remaining_folder_name = f"{label}_NOT_{conf_suffix}_{var_suffix}".replace("__", "_").strip("_")
        remaining_label_path = join(output_folder, remaining_folder_name)
        if not exists(remaining_label_path):
            mkdir(remaining_label_path)

    for batch in selected_batches:
        count += 1
        loadbar_save_dataset = 100 * count / max(1, len(selected_batches))

        batch_folder = batch[:-4]
        match = project_name + '.csv'
        if match in batch:
            match_size = len(match) + 1
            batch_folder = batch[:-match_size]

        temp_images_folder = join(images_folder, batch_folder)
        batch_images_folder = join(temp_images_folder, listdir(temp_images_folder)[0])

        csv_path = join(batches_folder, batch)
        df_temp = pd.read_csv(csv_path)
        print(f"Batch {batch}: {len(df_temp)} total rows")

        # Ensure needed columns exist
        if 'manual_conf' not in df_temp.columns:
            try:
                print(f"'manual_conf' missing in {batch}, updating...")
                df_temp = update_csv_file(csv_path)
            except Exception as e:
                print(f"Error updating manual_conf in {batch}: {e}")
                df_temp['manual_conf'] = 0.0

        if 'var_norm' not in df_temp.columns:
            df_temp['var_norm'] = 0.0

        df_temp['manual_conf'] = pd.to_numeric(df_temp['manual_conf'], errors='coerce').fillna(0.0)
        df_temp['var_norm'] = pd.to_numeric(df_temp['var_norm'], errors='coerce').fillna(0.0)

        df_filtered = df_temp.loc[df_temp['manual_label'].isin(selected_classes)]
        print(f"After class filtering: {len(df_filtered)} rows")

        filtered_rows = []
        remaining_rows = []

        for i, label in enumerate(selected_classes):
            conf_rng = default_conf
            if slider_values_conf is not None and i < len(slider_values_conf) and slider_values_conf[i] is not None:
                conf_rng = slider_values_conf[i]

            var_rng = default_var
            if slider_values_var is not None and i < len(slider_values_var) and slider_values_var[i] is not None:
                var_rng = slider_values_var[i]

            class_filtered = df_filtered[
                (df_filtered['manual_label'] == label) &
                (df_filtered['manual_conf'] >= conf_rng[0]) &
                (df_filtered['manual_conf'] < conf_rng[1]) &
                (df_filtered['var_norm'] >= var_rng[0]) &
                (df_filtered['var_norm'] < var_rng[1])
            ]
            filtered_rows.append(class_filtered)

            class_remaining = df_filtered[
                (df_filtered['manual_label'] == label) &
                ~(
                    (df_filtered['manual_conf'] >= conf_rng[0]) &
                    (df_filtered['manual_conf'] < conf_rng[1]) &
                    (df_filtered['var_norm'] >= var_rng[0]) &
                    (df_filtered['var_norm'] < var_rng[1])
                )
            ]
            remaining_rows.append(class_remaining)

        # Copy filtered images
        if filtered_rows:
            df_filtered_combined = pd.concat(filtered_rows, ignore_index=True)
            print(f"Copying {len(df_filtered_combined)} filtered images from batch {batch}")
            for _, row in df_filtered_combined.iterrows():
                name = row['names']
                label = row['manual_label']
                class_index = selected_classes.index(label)

                conf_rng = default_conf
                if slider_values_conf is not None and class_index < len(slider_values_conf) and slider_values_conf[class_index] is not None:
                    conf_rng = slider_values_conf[class_index]
                var_rng = default_var
                if slider_values_var is not None and class_index < len(slider_values_var) and slider_values_var[class_index] is not None:
                    var_rng = slider_values_var[class_index]

                conf_suffix = _make_range_folder_suffix("conf", conf_rng)
                var_suffix = _make_range_folder_suffix("var", var_rng)
                filtered_folder_name = f"{label}_{conf_suffix}_{var_suffix}".replace("__", "_").strip("_")
                filtered_label_path = join(output_folder, filtered_folder_name)

                copy2(join(batch_images_folder, name), join(filtered_label_path, name))

        # Copy remaining images
        if remaining_rows:
            df_remaining_combined = pd.concat(remaining_rows, ignore_index=True)
            print(f"Copying {len(df_remaining_combined)} remaining images from batch {batch}")
            for _, row in df_remaining_combined.iterrows():
                name = row['names']
                label = row['manual_label']
                class_index = selected_classes.index(label)

                conf_rng = default_conf
                if slider_values_conf is not None and class_index < len(slider_values_conf) and slider_values_conf[class_index] is not None:
                    conf_rng = slider_values_conf[class_index]
                var_rng = default_var
                if slider_values_var is not None and class_index < len(slider_values_var) and slider_values_var[class_index] is not None:
                    var_rng = slider_values_var[class_index]

                conf_suffix = _make_range_folder_suffix("conf", conf_rng)
                var_suffix = _make_range_folder_suffix("var", var_rng)
                remaining_folder_name = f"{label}_NOT_{conf_suffix}_{var_suffix}".replace("__", "_").strip("_")
                remaining_label_path = join(output_folder, remaining_folder_name)

                copy2(join(batch_images_folder, name), join(remaining_label_path, name))

    loadbar_save_dataset = 0
    return 0


def get_datasets_list():
    default_output_folder = join(getcwd(), 'output')
    if not exists(default_output_folder):
        makedirs(default_output_folder, exist_ok=True)
    datasets_list = [f for f in listdir(default_output_folder) if isdir(join(default_output_folder, f))]
    return datasets_list


projects_list = []
classes_per_batch = {}
loadbar_classes = 0
loadbar_save_dataset = 0
current_classes_list = []
previous_slider_values = []
previous_var_slider_values = []
get_projects_list()
datasets_list = get_datasets_list()


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}]
)


app.layout = html.Div([
    html.H1('ILT Data Explorer'),

    dcc.ConfirmDialog(id='confirm_save_dataset'),

    html.Datalist(
        id='list_suggested_datasets',
        children=[html.Option(value=name) for name in datasets_list]
    ),

    dbc.Container(
        dbc.Row([
            dbc.Col(dcc.Dropdown(projects_list, '', id='dropdown_project', clearable=False), width={"size": 12}),
        ]),
    ),

    dbc.Row([dbc.Col(html.Hr()), ], ),

    dbc.Container(
        dbc.Row([
            html.Div([
                dbc.Col(
                    [
                        html.Div([
                            dcc.Checklist(
                                options=[],
                                value=[],
                                id='checklist_batches',
                                labelStyle={'display': 'block'},
                                style={
                                    "overflow": "auto",
                                    'border': '1px solid #ddd',
                                    'borderRadius': '5px',
                                    'padding': '10px',
                                    'backgroundColor': '#f8f9fa',
                                    'height': '100%',
                                    'flexGrow': 1,
                                    'minHeight': 0
                                }
                            ),
                            html.Div([
                                dbc.Button('Update Classes & Histograms', n_clicks=0, id='button_update_classes',
                                           style={'background': 'steelblue', 'width': '100%'}),
                                dbc.Progress(value=0, id='progress_classes', style={'marginTop': '10px'})
                            ], style={'marginTop': '10px'})
                        ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'})
                    ], width=6
                ),
                dbc.Col([
                    html.Div([
                        html.Div(
                            id='slider-container',
                            style={
                                'height': '100%',
                                'overflow': 'auto',
                                'border': '1px solid #ddd',
                                'borderRadius': '5px',
                                'padding': '10px',
                                'backgroundColor': '#f8f9fa',
                                'flexGrow': 1,
                                'minHeight': 0
                            }
                        ),
                        html.Div([
                            dbc.Button('Update count', id='button_update_count', n_clicks=0,
                                       style={'background': 'seagreen', 'width': '100%'})
                        ], style={'marginTop': '10px'})
                    ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'})
                ], width=6
                ),
            ], style={'display': 'flex', 'height': '600px', 'gap': '15px', 'width': '100%'})
        ]),
    ),

    dbc.Row([dbc.Col(html.Hr()), ], ),

    dbc.Container(
        dbc.Row([
            dbc.Col(
                dbc.Input(
                    value='dataset',
                    id='input_save_dataset',
                    type="text",
                    style={'width': '100%', 'background': 'Floralwhite'},
                    list='list_suggested_datasets'
                ),
            ),
            dbc.Col([
                dbc.Button('Save Dataset', n_clicks=0, id='button_save_dataset',
                           style={'background': 'chocolate', 'width': '100%'}),
                dbc.Progress(value=0, id="progress_save_dataset"),
                dbc.Alert(children='', id="alert_save_dataset_problem", dismissable=True, is_open=False, color='warning',
                          duration=5000),
                dbc.Alert(children='', id="alert_save_dataset_success", dismissable=True, is_open=False),
            ])
        ])
    ),

    dcc.Interval(id='clock', interval=1000, n_intervals=0, max_intervals=-1),
    dcc.Store(id='button_display_classes_nclicks', data=0),
    dcc.Store(id='button_update_classes_nclicks', data=0),
    dcc.Store(id='dataset_name_ok', data=False),
    dcc.Store(id='save_dataset_result', data=0),
])


@app.callback(
    Output("progress_save_dataset", "value"),
    Input("clock", "n_intervals"))
def progress_save_dataset_update(n):
    global loadbar_save_dataset
    return (loadbar_save_dataset,)


@app.callback(
    Output("progress_classes", "value"),
    Input("clock", "n_intervals"))
def progress_classes_update_separate(n):
    global loadbar_classes
    return (loadbar_classes,)


@app.callback(
    Output('slider-container', 'children'),
    Output('button_display_classes_nclicks', 'data'),
    Output('button_update_classes_nclicks', 'data'),
    Input('button_update_classes', 'n_clicks'),
    Input('button_update_count', 'n_clicks'),
    State('slider-container', 'children'),
    State({'type': 'dynamic-slider', 'index': dash.ALL}, 'value'),          # manual_conf sliders
    State({'type': 'dynamic-slider-var', 'index': dash.ALL}, 'value'),      # var_norm sliders
    State('dropdown_project', 'value'),
    State('checklist_batches', 'value'),
    State('button_display_classes_nclicks', 'data'),
    State('button_update_classes_nclicks', 'data')
)
def update_classes_list(button_update_classes_clicks, button_update_count_clicks, slide_container,
                        slider_values_conf, slider_values_var,
                        project_name, checklist_value,
                        prev_nclicks, prev_button_clicks):
    import dash
    global current_classes_list, previous_slider_values, previous_var_slider_values, loadbar_classes
    components = []

    print("=== update_classes_list called ===")
    print(f"button_update_classes_clicks: {button_update_classes_clicks}")
    print(f"button_update_count_clicks: {button_update_count_clicks}")
    print(f"checklist_value: {checklist_value}")
    print(f"slider_values_conf: {slider_values_conf}")
    print(f"slider_values_var: {slider_values_var}")
    print(f"project_name: {project_name}")

    ctx = dash.callback_context
    button_triggered = False
    trigger_id = None

    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'button_update_classes' and button_update_classes_clicks and button_update_classes_clicks > prev_button_clicks:
            button_triggered = True
        elif trigger_id == 'button_update_count' and button_update_count_clicks and button_update_count_clicks > 0:
            button_triggered = True

    if button_triggered:
        batches_folder = join(getcwd(), 'main', 'assets', project_name, 'dataframes')

        if checklist_value:
            for batch_file in checklist_value:
                file_path = join(batches_folder, batch_file)
                try:
                    df = pd.read_csv(file_path, nrows=1)
                    if 'manual_conf' not in df.columns:
                        print(f"'manual_conf' missing in {batch_file}, updating...")
                        df = update_csv_file(file_path)
                except Exception as e:
                    print(f"Error reading {batch_file}: {e}")

        classes_list = get_classes(batches_folder, checklist_value)

        # Recompute counts using BOTH sliders (manual_conf AND var_norm)
        get_classes_per_batch(
            batches_folder, checklist_value, classes_list,
            slider_values_conf=slider_values_conf,
            slider_values_var=slider_values_var,
            changed_slider_index=None
        )

        classes_text, classes_list = update_classes_project(checklist_value, project_name)

        # Histogram data for both metrics
        hist_data_conf = get_histogram_data_metric(batches_folder, checklist_value, classes_list, metric_col='manual_conf')
        hist_data_var = get_histogram_data_metric(batches_folder, checklist_value, classes_list, metric_col='var_norm')

        current_classes_list = classes_list

        if slider_values_conf:
            previous_slider_values = slider_values_conf.copy()
        if slider_values_var:
            previous_var_slider_values = slider_values_var.copy()

        for i in range(len(classes_text)):
            # Defaults
            slider_conf_value = slider_values_conf[i] if slider_values_conf and i < len(slider_values_conf) and slider_values_conf[i] is not None else [0.85, 1.0]
            slider_var_value = slider_values_var[i] if slider_values_var and i < len(slider_values_var) and slider_values_var[i] is not None else [0.0, 1.0]

            # --- Histogram for manual_conf ---
            hist_fig_conf = go.Figure()
            if classes_list[i] in hist_data_conf:
                hist_fig_conf.add_trace(go.Histogram(
                    x=hist_data_conf[classes_list[i]],
                    xbins=dict(start=0, end=1.001, size=0.05),
                    opacity=0.7,
                    nbinsx=20
                ))
                hist_fig_conf.add_vline(x=slider_conf_value[0], line_dash="dash", line_color="red")
                hist_fig_conf.add_vline(x=slider_conf_value[1], line_dash="dash", line_color="red")

            hist_fig_conf.update_layout(
                height=48,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                xaxis=dict(showticklabels=False, range=[-0.07, 1.07], showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
            )

            histogram_conf = dcc.Graph(
                id={'type': 'dynamic-histogram', 'index': i},
                figure=hist_fig_conf,
                config={'displayModeBar': False}
            )

            slider_conf = dcc.RangeSlider(
                id={'type': 'dynamic-slider', 'index': i},
                min=0,
                max=1.0,
                step=0.05,
                value=slider_conf_value,
                marks={0: '0', 1: '1'},
                tooltip={
                    "placement": "bottom",
                    "always_visible": True,
                    "style": {"color": "DarkBlue", "fontSize": "16px"},
                }
            )

            # --- Histogram for var_norm ---
            hist_fig_var = go.Figure()
            if classes_list[i] in hist_data_var:
                hist_fig_var.add_trace(go.Histogram(
                    x=hist_data_var[classes_list[i]],
                    xbins=dict(start=0, end=1.001, size=0.05),
                    opacity=0.7,
                    nbinsx=20
                ))
                hist_fig_var.add_vline(x=slider_var_value[0], line_dash="dash", line_color="red")
                hist_fig_var.add_vline(x=slider_var_value[1], line_dash="dash", line_color="red")

            hist_fig_var.update_layout(
                height=48,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                xaxis=dict(showticklabels=False, range=[-0.07, 1.07], showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
            )

            histogram_var = dcc.Graph(
                id={'type': 'dynamic-histogram-var', 'index': i},
                figure=hist_fig_var,
                config={'displayModeBar': False}
            )

            slider_var = dcc.RangeSlider(
                id={'type': 'dynamic-slider-var', 'index': i},
                min=0,
                max=1.0,
                step=0.05,
                value=slider_var_value,
                marks={0: '0', 1: '1'},
                tooltip={
                    "placement": "bottom",
                    "always_visible": True,
                    "style": {"color": "DarkBlue", "fontSize": "16px"},
                }
            )

            checklist = dcc.Checklist(
                id={'type': 'dynamic-checklist', 'index': i},
                options=[classes_text[i]],
                value=['enable'],
                style={'marginLeft': '0px', 'width': '25%'}
            )

            conf_block = html.Div([
                html.Div("manual_conf", style={"fontSize": "12px", "color": "#555"}),
                histogram_conf,
                html.Div(slider_conf, style={'marginTop': '-16px'})
            ], style={'width': '35%'})

            var_block = html.Div([
                html.Div("var_norm", style={"fontSize": "12px", "color": "#555"}),
                histogram_var,
                html.Div(slider_var, style={'marginTop': '-16px'})
            ], style={'width': '35%'})

            row = html.Div(
                [checklist, conf_block, var_block],
                style={
                    'display': 'flex',
                    'alignItems': 'flex-start',
                    'gap': '15px',
                    'marginBottom': '15px'
                }
            )

            loadbar_classes = 80 + 20 * i / max(1, len(classes_text))
            components.append(row)

        loadbar_classes = 100

        if trigger_id == 'button_update_classes':
            return components, prev_nclicks, button_update_classes_clicks
        else:
            return components, prev_nclicks + 1, prev_button_clicks

    return components, prev_nclicks, prev_button_clicks


@app.callback(
    Output('checklist_batches', 'options'),
    Output('checklist_batches', 'value'),
    Output('input_save_dataset', 'value'),
    Input('dropdown_project', 'value')
)
def update_batches_list(project_name):
    global classes_per_batch, loadbar_classes, loadbar_save_dataset, previous_slider_values, previous_var_slider_values

    if project_name != '':
        batches_folder = join(getcwd(), 'main', 'assets', project_name, 'dataframes')
        batches_list = [f for f in listdir(batches_folder) if f[-4:] == '.csv']
        batches_list.sort()

        batches_text = []
        for i in range(len(batches_list)):
            batches_text.append({'label': ' ' + str(i + 1) + ': ' + batches_list[i], 'value': batches_list[i]})

        classes_per_batch = {}
        loadbar_classes = 0
        loadbar_save_dataset = 0
        previous_slider_values = []
        previous_var_slider_values = []

        return batches_text, batches_list, project_name + '_dataset'
    return [], [], 'dataset'


@app.callback(
    Output('alert_save_dataset_success', 'children'),
    Output('alert_save_dataset_success', 'is_open'),
    Input('confirm_save_dataset', 'submit_n_clicks'),
    State('dropdown_project', 'value'),
    State('input_save_dataset', 'value'),
    State('checklist_batches', 'value'),
    State({'type': 'dynamic-checklist', 'index': dash.ALL}, 'value'),
    State({'type': 'dynamic-slider', 'index': dash.ALL}, 'value'),       # manual_conf
    State({'type': 'dynamic-slider-var', 'index': dash.ALL}, 'value'),   # var_norm
)
def save_dataset_confirmed(nclicks, project_name, dataset_name, selected_batches, selected_classes, slider_values_conf, slider_values_var):
    global current_classes_list
    print("Save dataset confirmed")
    print()
    if nclicks:
        actual_selected_classes = []
        if selected_classes and current_classes_list:
            for i, class_list in enumerate(selected_classes):
                if class_list and i < len(current_classes_list):
                    actual_selected_classes.append(current_classes_list[i])

        default_output_folder = join(getcwd(), 'output')
        output_folder = join(default_output_folder, dataset_name)

        save_dataset(output_folder, project_name, actual_selected_classes, selected_batches, slider_values_conf, slider_values_var)
        return dataset_name + ' saved successfully', True
    return '', False


@app.callback(
    Output('confirm_save_dataset', 'message'),
    Output('confirm_save_dataset', 'displayed'),
    Input('dataset_name_ok', 'data'),
    State('input_save_dataset', 'value'),
)
def save_dataset_confirmation(name_ok, dataset_name):
    default_output_folder = join(getcwd(), 'output')
    output_folder = join(default_output_folder, dataset_name)

    if name_ok is True:
        if exists(output_folder):
            return dataset_name + ' already exists. Do you really want to merge these images into the existing dataset?', True
        else:
            return dataset_name + ' will be created. Proceed?', True
    return '', False


@app.callback(
    Output('alert_save_dataset_problem', 'children'),
    Output('alert_save_dataset_problem', 'is_open'),
    Output('dataset_name_ok', 'data'),
    Input('button_save_dataset', 'n_clicks'),
    State('dropdown_project', 'value'),
    State('checklist_batches', 'value'),
    State({'type': 'dynamic-checklist', 'index': dash.ALL}, 'value'),
)
def click_save_dataset(nclicks, project_name, selected_batches, selected_classes):
    global current_classes_list
    if nclicks > 0:
        default_output_folder = join(getcwd(), 'output')
        if not exists(default_output_folder):
            mkdir(default_output_folder)

        actual_selected_classes = []
        if selected_classes and current_classes_list:
            for i, class_list in enumerate(selected_classes):
                if class_list and i < len(current_classes_list):
                    actual_selected_classes.append(current_classes_list[i])

        if project_name == '':
            return 'No project was selected', True, False
        elif len(selected_batches) == 0:
            return 'No batches were selected', True, False
        elif len(actual_selected_classes) == 0:
            return 'No classes were selected', True, False
        else:
            return '', False, True
    return '', False, False


port = 8030
opened = False

if not opened:
    webbrowser.open('http://127.0.0.1:' + str(port) + '/', new=2, autoraise=True)
    opened = True

if __name__ == '__main__':
    app.title = 'ILT Data Explorer'
    app.run_server(debug=False, port=port)

