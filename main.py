import time
import dearpygui.dearpygui as dpg
import keras
import numpy as np
from PIL import Image as im
from keras.models import model_from_json


def loadNetwork():
    t0 = time.time()
    # load json and create model
    with open("model.json", 'r') as json_file:  # with statement automatically closes file afterwards
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("model.h5")

    loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # print(f"loaded network in {time.time() - t0}s")
    return loaded_model


def main():
    loaded_model = loadNetwork()

    dpg.create_context()

    texture_res = 28
    texture_data = [1, 1, 1, 1] * (texture_res ** 2)

    graph_values_x = []
    for i in range(0, 10):
        graph_values_x.append(i)

    def runNetwork():
        # convert 1D array to 3D array
        imagedata = np.array(texture_data).reshape(28, 28, 4)

        # convert 3D array into image and convert from RGBA to greyscale (L)
        imagedata = im.fromarray(imagedata, "RGBA").convert("L")

        # Convert greyscale image back to array of shape (28, 28, 1)
        imagedata = keras.utils.img_to_array(imagedata)

        # Insert fourth axis at the first position of the vector, resulting in the shape (1, 28, 28, 1)
        imagedata = np.expand_dims(imagedata, 0)

        t0 = time.time()
        prediction_values = loaded_model.predict(imagedata)  # Returns list of predictions for each digit.

        max_value = np.max(prediction_values)  # Find maximum value
        max_index = np.where(prediction_values == max_value).index(1)  # Find max_value's index
        dpg.set_value("text_output", f"Guessed digit: {max_index}")  # Each index corresponds to a digit.
        dpg.set_value("text_duration", f"Time to calculate: {round((time.time() - t0) * 1000, 2)}ms")
        dpg.set_value('line_series', [graph_values_x, prediction_values * [100]])

    def clearCanvas():
        print("Clearing canvas")
        dpg.set_value("text_output", "Guessed digit: NONE")
        dpg.set_value("text_duration", "Time to calculate: 0.00ms")
        dpg.set_value('line_series', [graph_values_x, [0] * 10])
        nonlocal texture_data
        texture_data = [1, 1, 1, 1] * (28 ** 2)

    # def mouseInRect(drawlist, canvas=None):
    #     px, py = dpg.get_mouse_pos(local=False)
    #     for item in drawlist:
    #         if (drawable := item.contains(px, py, canvas=canvas)) is not None:
    #             return drawable
    #     return None
    #
    # def mouseClick(x_coord_entry, canvas):
    #     px, py = dpg.get_mouse_pos(local=False)
    #     if px < 200:
    #         return
    #
    #     global currentSelection
    #     global currentPick
    #     if currentSelection is None:
    #         if currentPick is not None:
    #             dpg.delete_item(currentPick)
    #         currentPick = None
    #         return
    #
    #     selected = dpg.get_item_user_data(currentSelection)
    #     config = dpg.get_item_configuration(selected)
    #     pmin = [config['pmin'][0] - 2, config['pmin'][1] - 2]
    #     pmax = [config['pmax'][0] + 2, config['pmax'][1] + 2]
    #     dpg.set_value(x_coord_entry, config['pmin'][0])
    #
    #     if currentPick is None:
    #         currentPick = dpg.draw_rectangle(pmin, pmax, thickness=2, parent=canvas, color=[255, 0, 0, 255],
    #                                          user_data=selected)
    #
    #     if dpg.get_item_user_data(currentPick) != selected:
    #         dpg.configure_item(currentPick, pmin=pmin, pmax=pmax, user_data=selected)
    #
    # def mouseMove(drawlist, canvas):
    #     global currentSelection
    #     if (selected := mouseInRect(drawlist, canvas)) is None:
    #         if currentSelection is not None:
    #             dpg.delete_item(currentSelection)
    #         currentSelection = None
    #         return
    #
    #     config = dpg.get_item_configuration(selected)
    #     pmin = [config['pmin'][0] - 2, config['pmin'][1] - 2]
    #     pmax = [config['pmax'][0] + 2, config['pmax'][1] + 2]
    #
    #     if currentSelection is None:
    #         currentSelection = dpg.draw_rectangle(pmin, pmax, thickness=2, parent=canvas, user_data=selected)
    #
    #     if dpg.get_item_user_data(currentSelection) != selected:
    #         dpg.configure_item(currentSelection, pmin=pmin, pmax=pmax, user_data=selected)
    #
    # def mouseDrag():
    #     global currentPick
    #     if currentPick is None:
    #         return
    #
    #     px, py = dpg.get_mouse_pos(local=False)
    #     if px < 200:
    #         return
    #
    #     selected = dpg.get_item_user_data(currentPick)
    #     config = dpg.get_item_configuration(selected)
    #     pmin = [px, py]
    #     w = config['pmax'][0] - config['pmin'][0]
    #     h = config['pmax'][1] - config['pmin'][1]
    #     pmax = [px + w, py + h]
    #     dpg.configure_item(selected, pmin=pmin, pmax=pmax)
    #     dpg.configure_item(currentPick, pmin=pmin, pmax=pmax)
    #
    # with dpg.handler_registry():
    #     dpg.add_mouse_click_handler(callback=lambda s, d: mouseClick(x_coord_entry, canvas, ))
    #     dpg.add_mouse_move_handler(callback=lambda s, d: mouseMove(drawlist, canvas, ))
    #     dpg.add_mouse_drag_handler(callback=lambda s, d: mouseDrag())

    with dpg.texture_registry():  # show=True):
        dpg.add_dynamic_texture(width=28, height=28, default_value=texture_data, tag="texture_tag")

    with dpg.window(tag="Primary Window"):
        with dpg.group(horizontal=True):
            with dpg.group(horizontal=False):
                with dpg.group(horizontal=True):
                    dpg.add_text("Draw on canvas below.")
                    dpg.add_button(label="Clear canvas", callback=clearCanvas)
                dpg.add_image(texture_tag="texture_tag", width=28 * 4, height=28 * 4)
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
                    graph_values_x,
                    [0] * 10,
                    label="Prediction",
                    parent="y_axis",
                    tag="line_series"
                )

                # limits
                dpg.set_axis_limits("x_axis", -0.5, 9.5)
                dpg.set_axis_limits("y_axis", 0, 100)
    dpg.set_primary_window("Primary Window", True)

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
