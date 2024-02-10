import os
import time
from random import random

import dearpygui.dearpygui as dpg
import keras
import numpy as np
from PIL import Image as im, ImageDraw
from keras.models import model_from_json

NETWORK_RESOLUTION = 28


def loadTestImages():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255
    x_test = np.expand_dims(x_test, -1)
    return x_test


def loadNetwork():
    # load json and create model
    with open("model.json", 'r') as json_file:  # "with" statement automatically closes file afterwards
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return loaded_model


def main():
    dpg.create_context()

    loaded_model = loadNetwork()
    test_images = loadTestImages()
    img = im.new("L", (NETWORK_RESOLUTION, NETWORK_RESOLUTION), 0)
    draw = ImageDraw.Draw(img)
    mouse_dxp = mouse_dyp = 0
    drawing = False
    graph_values_x = np.arange(10)  # Create list from 0 to 9

    def runNetwork():
        # Convert greyscale image back to array of shape (28, 28, 1)
        processed_texture = np.asarray(img.resize((NETWORK_RESOLUTION, NETWORK_RESOLUTION)))

        # Insert fourth axis at the first position of the vector, resulting in the shape (1, 28, 28, 1)
        expanded_texture = np.expand_dims(processed_texture, 0)

        t0 = time.time()

        # Returns list of predictions for each digit.
        prediction_values = loaded_model(expanded_texture) # loaded_model.predict(expanded_texture)
        print(time.time() - t0)

        max_value = np.max(prediction_values)  # Find maximum value
        max_index = np.where(prediction_values == max_value)[1][0]  # Find max_value's index
        dpg.set_value("text_output", f"Geraden cijfer: {max_index}")  # Each index corresponds to a digit.
        dpg.set_value("text_duration", f"Berekentijd: {round((time.time() - t0) * 1000, 2)}ms")
        dpg.set_value('line_series', [graph_values_x, prediction_values * [100]])

    def updateTexture():
        texture_data = np.asarray(img.convert("RGBA"))
        dpg.set_value("texture_tag", texture_data)

    def clearCanvas(redraw=True):
        if redraw:
            draw.rectangle((0, 0, img.width, img.height), 0)
            updateTexture()
        dpg.set_value("text_output", "Geraden cijfer: nog niets")
        dpg.set_value("text_duration", "Berekentijd: 0.00ms")
        dpg.set_value('line_series', [graph_values_x, [0] * 10])

    def pickRandomImage():
        clearCanvas(redraw=False)
        image_index = int(random() * len(test_images) + 0.5)
        test_image_tensor = test_images[image_index] ** 0.5 * 2  # 2x multiplier so it shows up better
        test_image_flat = test_image_tensor.flatten().flatten()
        img.putdata(test_image_flat)
        updateTexture()
        runNetwork()

    def drawLineOnImage(from_pos: (int, int), to_pos: (int, int)):
        width = img.height / 8
        width_half = width / 4
        draw.line(xy=[from_pos, to_pos], fill=1, width=int(width), joint="curve")
        draw.ellipse(xy=[(from_pos[0] - width_half, from_pos[1] - width_half),
                         (from_pos[0] + width_half, from_pos[1] + width_half)], fill=1, width=0)
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

    def mouseClick():
        nonlocal drawing
        if not mouseOnCanvas():
            drawing = False
            return
        drawing = True

        nonlocal mouse_dxp, mouse_dyp
        mouse_dxp, mouse_dyp = getMouseCanvasPos()

    def mouseRelease():
        nonlocal drawing
        if not drawing:
            return
        drawing = False
        runNetwork()

    def mouseDrag():
        if not drawing:
            return

        nonlocal mouse_dxp, mouse_dyp
        mouse_dx, mouse_dy = getMouseCanvasPos()
        drawLineOnImage((mouse_dxp, mouse_dyp), (mouse_dx, mouse_dy))

        mouse_dxp, mouse_dyp = mouse_dx, mouse_dy
        runNetwork()

    with dpg.handler_registry():
        dpg.add_mouse_click_handler(callback=mouseClick)
        dpg.add_mouse_release_handler(callback=mouseRelease)
        dpg.add_mouse_drag_handler(callback=mouseDrag)

    def resized():
        width = dpg.get_viewport_width()
        height = dpg.get_viewport_height()
        print(f"resized width: {width} height: {height}")

    with dpg.item_handler_registry(tag="#resize_handler"):
        dpg.add_item_resize_handler(callback=resized)

    with dpg.texture_registry():  # show=True):
        dpg.add_dynamic_texture(width=img.width, height=img.height,
                                default_value=[1, 1, 1, 1] * (img.width * img.height), tag="texture_tag")

    # TODO
    #  - Meer contrast tussen interface elementen en achtergrond
    #  - 3D effect knoppen?
    #  - Het geraadde getal rechts van het canvas tonen, met dezelfde grootte als het canvas.
    #  - Veel grotere letters
    #  - Grafieken en uitvoertijd zijn details, hou hun klein
    #  - Maak het aantrekkelijk

    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(target=dpg.mvStyleVar_FramePadding, x=7, y=13)
    dpg.bind_theme(global_theme)

    with dpg.window(tag="Primary Window"):
        with dpg.group(horizontal=True):
            with dpg.group(horizontal=False, width=325, tag="width0"):
                dpg.add_text("Teken op het onderstaande canvas.")
                dpg.add_image(texture_tag="texture_tag", width=325, height=325, tag="canvas_image")
                # with dpg.group(horizontal=True):
                # dpg.add_button(label="Start / test", callback=runNetwork, width=-1)
                dpg.add_button(label="Leeg canvas", callback=clearCanvas, width=-1)
                dpg.add_button(label="Willekeurig cijfer", callback=pickRandomImage, width=-1)
                dpg.add_text(default_value="Geraden cijfer: nog niets", tag="text_output")
                dpg.add_text(default_value="Berekentijd: 0.00ms", tag="text_duration")

            # create plot
            with dpg.plot(label="Cijfer voorspellingen", height=-1, width=-1):
                dpg.add_plot_axis(dpg.mvXAxis, label="Cijfers", tag="x_axis", no_gridlines=True, no_tick_marks=True)
                dpg.set_axis_ticks("x_axis", (
                    ("0", 0), ("1", 1), ("2", 2), ("3", 3), ("4", 4), ("5", 5), ("6", 6), ("7", 7), ("8", 8), ("9", 9)))

                dpg.add_plot_axis(dpg.mvYAxis, label="Voorspellingen(%)", tag="y_axis")
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

    dpg.set_viewport_resize_callback(resized)
    dpg.create_viewport(
        title='Artificial Computer Neural Example (ACNE)',
        width=650,
        height=550,
        # resizable=False,
        min_width=600,
        min_height=550,

        small_icon="CustomIcon.ico",
        large_icon="CustomIcon.ico"
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == '__main__':  # Will not run when file is loaded as import
    main()
