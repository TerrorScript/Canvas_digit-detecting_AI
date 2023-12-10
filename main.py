import time

import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image as im, ImageDraw
from keras.models import model_from_json

NETWORK_RESOLUTION = 28


def loadNetwork():
    # load json and create model
    with open("model.json", 'r') as json_file:  # "with" statement automatically closes file afterwards
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return loaded_model


def interface():
    # Variables
    loaded_model = loadNetwork()
    img = im.new("L", (70, 70), 1)
    draw = ImageDraw.Draw(img)
    mouse_dxp = mouse_dyp = 0
    drawing = False
    graph_values_x = np.arange(10)  # Create list from 0 to 9

    # Functions
    def runNetwork():
        # Convert greyscale image back to array of shape (28, 28, 1)
        processed_texture = np.asarray(img.resize((NETWORK_RESOLUTION, NETWORK_RESOLUTION)))

        # Insert fourth axis at the first position of the vector, resulting in the shape (1, 28, 28, 1)
        expanded_texture = np.expand_dims(processed_texture, 0)

        t0 = time.time()
        prediction_values = loaded_model.predict(expanded_texture)  # Returns list of predictions for each digit.

        max_value = np.max(prediction_values)  # Find maximum value
        print(max_value)
        print(np.where(prediction_values == max_value)[0])
        max_index = np.where(prediction_values == max_value)[0]  # Find max_value's index
        dpg.set_value("text_output", f"Guessed digit: {max_index}")  # Each index corresponds to a digit.
        dpg.set_value("text_duration", f"Time to calculate: {round((time.time() - t0) * 1000, 2)}ms")
        dpg.set_value('line_series', [graph_values_x, prediction_values * [100]])

    def updateTexture():
        texture_data = np.asarray(img.convert("RGBA"))
        dpg.set_value("texture_tag", texture_data)

    def clearCanvas():
        draw.rectangle((0, 0, img.width, img.height), 1)
        updateTexture()
        dpg.set_value("text_output", "Guessed digit: NONE")
        dpg.set_value("text_duration", "Time to calculate: 0.00ms")
        dpg.set_value('line_series', [graph_values_x, [0] * 10])

    def drawLineOnImage(from_pos: (int, int), to_pos: (int, int)):
        draw.line([from_pos, to_pos], 0, 15)
        updateTexture()

    def mouseOnCanvas():
        return dpg.is_item_hovered('canvas_image')

    def getMouseCanvasPos() -> (int, int):
        item_size = dpg.get_item_rect_size("canvas_image")
        scale_x = img.width / item_size[0]
        scale_y = img.height / item_size[1]

        item_pos = dpg.get_item_pos("canvas_image")
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        return (mouse_x - item_pos[0]) * scale_x, (mouse_y - item_pos[1]) * scale_y

    def mouseClick(x_coord_entry, canvas):
        nonlocal drawing
        if not mouseOnCanvas():
            drawing = False
            return
        drawing = True

        nonlocal mouse_dxp, mouse_dyp
        mouse_dxp, mouse_dyp = getMouseCanvasPos()

    def mouseDrag():
        print("Check mouse drag")
        if not drawing:
            return
        print("Check mouse drag success")

        nonlocal mouse_dxp, mouse_dyp
        mouse_dx, mouse_dy = getMouseCanvasPos()
        drawLineOnImage((mouse_dxp, mouse_dyp), (mouse_dx, mouse_dy))

        mouse_dxp, mouse_dyp = mouse_dx, mouse_dy

    with dpg.handler_registry():
        dpg.add_mouse_click_handler(callback=mouseClick)
        dpg.add_mouse_drag_handler(callback=mouseDrag)

    with dpg.texture_registry():  # show=True):
        dpg.add_dynamic_texture(width=img.width, height=img.height,
                                default_value=[1, 1, 1, 1] * (img.width * img.height), tag="texture_tag")

    with dpg.window(tag="Primary Window"):
        with dpg.group(horizontal=True):
            with dpg.group(horizontal=False):
                with dpg.group(horizontal=True):
                    dpg.add_text("Draw on canvas below.")
                    dpg.add_button(label="Clear canvas", callback=clearCanvas)
                dpg.add_image(texture_tag="texture_tag", width=img.width * 2, height=img.height * 2, tag="canvas_image")
                dpg.add_button(label="Run network", callback=runNetwork)
                dpg.add_text(default_value="Guessed digit: NONE", tag="text_output")
                dpg.add_text(default_value="Time to calculate: 0.00ms", tag="text_duration")

            # create plot
            with dpg.plot(label="Digit predictions", height=-1, width=-1):
                # dpg.add_plot_legend()  # Optional

                dpg.add_plot_axis(dpg.mvXAxis, label="Digits", tag="x_axis", no_gridlines=True, no_tick_marks=True)
                dpg.set_axis_ticks("x_axis", (
                    ("0", 0),
                    ("1", 1),
                    ("2", 2),
                    ("3", 3),
                    ("4", 4),
                    ("5", 5),
                    ("6", 6),
                    ("7", 7),
                    ("8", 8),
                    ("9", 9)
                ))

                dpg.add_plot_axis(dpg.mvYAxis, label="Predictions(%)", tag="y_axis")

                dpg.add_bar_series(
                    graph_values_x.tolist(),
                    [0] * 10,
                    label="Prediction",
                    parent="y_axis",
                    tag="line_series"
                )

                # limits
                dpg.set_axis_limits("x_axis", -0.5, 9.5)
                dpg.set_axis_limits("y_axis", 0, 100)

    updateTexture()
    dpg.set_primary_window("Primary Window", True)
    # dpg.show_debug()


def main():
    dpg.create_context()

    interface()

    dpg.create_viewport(
        title='Canvas digit-detecting AI',
        width=600,
        height=300,
        # resizable=False,
        min_width=500,
        min_height=250
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == '__main__':  # Will not run when file is loaded as import
    main()
