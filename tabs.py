import PySimpleGUI as sg

# Define the layout of your tabs
tab1_layout = [
    [sg.Text("This is the content of Tab 1")],
    [sg.Button("Click me")]
]

tab2_layout = [
    [sg.Text("This is the content of Tab 2")],
    [sg.InputText(), sg.Button("Submit")]
]

# Create the tabs
tabs = [
    [sg.Tab('Tab 1', tab1_layout)],
    [sg.Tab('Tab 2', tab2_layout)]
]

# Create the main layout with the TabGroup
layout = [
    [sg.TabGroup(tabs, enable_events=True, key='-TABGROUP-')],
    [sg.Button("Exit")]
]

# Create the window
window = sg.Window("Tab Example", layout, resizable=True)

# Event loop
while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED or event == "Exit":
        break

    # Handle tab change event
    if event == '-TABGROUP-':
        selected_tab = values['-TABGROUP-']
        print(f"Switched to tab: {selected_tab}")

# Close the window
window.close()
