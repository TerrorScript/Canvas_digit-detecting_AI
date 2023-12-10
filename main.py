import time
from datetime import datetime

import dearpygui.dearpygui as dpg
import keras
import numpy as np
from PIL import Image as im, Image, ImageDraw
from keras.models import model_from_json


def open_drawing_window(filetype, title, size_h_w: tuple = None):
    # Based off of https://www.reddit.com/r/DearPyGui/comments/rpj1b0/dpg_touchscreen_drawing_with_pen_doesnt_work/
    # TODO
    #  After some analysis:
    #  1. Define list tracking all drawn points/lines
    #  2. Create drawlist to display all of those points/lines
    #  3. Bind to mouse events
    #    3.1. Update tracking list
    #    3.2. Update drawlist
    #  4. Process for network
    #    4.1. Create blank image (using Image.new)
    #    4.2. Create draw object (using ImageDraw.Draw) to modify blank image directly
    #    4.3. Loop over points/lines and apply operations to blank image using draw object
    #    4.4. Process now-modified image
    #    4.5. Feed now-processed image into network

    drawbox_width = size_h_w
    drawbox_height = size_h_w

    points_list = []
    tmp_points_list = []

    with dpg.handler_registry(show=True, tag="__demo_mouse_handler") as draw_mouse_handler:
        m_wheel = dpg.add_mouse_wheel_handler()
        m_click = dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left)
        m_double_click = dpg.add_mouse_double_click_handler(button=dpg.mvMouseButton_Left)
        m_release = dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left)
        m_drag = dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, threshold=0.0000001)
        m_down = dpg.add_mouse_down_handler(button=dpg.mvMouseButton_Left)
        m_move = dpg.add_mouse_move_handler()

    def _event_handler(sender, data):
        event_type = dpg.get_item_info(sender)["type"]

        if event_type == "mvAppItemType::mvMouseReleaseHandler":
            print("---------")
            if dpg.is_item_hovered('draw_canvas'):
                points_list.append(tmp_points_list[:])
                # print('master list, len', len(points_list), points_list)
                if dpg.does_item_exist(item="drawn_lines_layer"):
                    dpg.delete_item(item="drawn_lines_layer")
                if dpg.does_item_exist(item="drawn_lines_layer_tmp"):
                    dpg.delete_item(item="drawn_lines_layer_tmp")
                dpg.add_draw_layer(tag="drawn_lines_layer", parent=canvas)
                for x in points_list:
                    # print('sublist, len', len(x), x)
                    dpg.draw_polyline(points=x,
                                      parent="drawn_lines_layer",
                                      closed=False,
                                      color=(175, 115, 175, 255),
                                      thickness=2)
                tmp_points_list.clear()

        elif event_type == "mvAppItemType::mvMouseDownHandler" or event_type == "mvAppItemType::mvMouseDragHandler":
            if dpg.is_item_hovered('draw_canvas'):
                cur_mouse_pos = dpg.get_drawing_mouse_pos()
                tmp_points_list.append(tuple(cur_mouse_pos))
                if dpg.does_item_exist(item="drawn_lines_layer_tmp"):
                    dpg.delete_item(item="drawn_lines_layer_tmp")
                if dpg.does_item_exist(item="drawn_lines_layer_tmp"):
                    dpg.delete_item(item="drawn_lines_layer_tmp")
                dpg.add_draw_layer(tag="drawn_lines_layer_tmp", parent=canvas)
                dpg.draw_polyline(points=tmp_points_list,
                                  parent="drawn_lines_layer_tmp",
                                  closed=False,
                                  color=(175, 115, 175, 255),
                                  thickness=2)

    with dpg.window(label="Drawing window", no_close=True, modal=True, tag="draw_window"):
        def erase(sender, data):
            if sender == 'erase_last':
                if points_list:
                    points_list.pop()
                    if dpg.does_item_exist(item="drawn_lines_layer"):
                        dpg.delete_item(item="drawn_lines_layer")

                    dpg.add_draw_layer(tag="drawn_lines_layer", parent=canvas)
                    for x in points_list:
                        dpg.draw_polyline(points=x,
                                          parent="drawn_lines_layer",
                                          closed=False,
                                          color=(175, 115, 175, 255),
                                          thickness=2)
                else:
                    pass

            elif sender == 'erase_all':
                points_list.clear()
                if dpg.does_item_exist(item="drawn_lines_layer"):
                    dpg.delete_item(item="drawn_lines_layer")

        def save_n_close(sender, data):
            if sender == "save_close":
                output_img = Image.new(mode="RGB", size=(drawbox_width, drawbox_height))
                draw = ImageDraw.Draw(output_img)
                for y in points_list:
                    draw.line(y, None, 2, None)
                output_img.save('{type}_{title}_{date}.png'.format(type=filetype,
                                                                   title=title,
                                                                   date=datetime.now().strftime("%Y_%m_%d-%H_%M_%S")))

            dpg.delete_item("draw_window")
            dpg.configure_item(item=draw_mouse_handler, show=False)

            if __name__ == '__main__':
                pass
                # dpg.stop_dearpygui()

        for handler in dpg.get_item_children("__demo_mouse_handler", 1):
            dpg.set_item_callback(handler, _event_handler)

        with dpg.group(tag='cnt_btns', horizontal=True, parent="draw_window") as buttons:
            dpg.add_button(label='Erase last', callback=erase, tag='erase_last')
            dpg.add_spacer(width=30)
            dpg.add_button(label='Erase all', callback=erase, tag='erase_all')
            dpg.add_spacer(width=30)
            dpg.add_button(label='Save and close', callback=save_n_close, tag='save_close')
            dpg.add_spacer(width=30)
            dpg.add_button(label='Close without saving', callback=save_n_close, tag='close_no_save')

        dpg.add_text(default_value="Please sign in the box below", parent='draw_window')

        with dpg.child_window(label="canvas_border", tag='canvas_border', width=drawbox_width + 10,
                              height=drawbox_height + 10, border=True, no_scrollbar=True, parent='draw_window'):
            with dpg.drawlist(width=drawbox_width, height=drawbox_height,
                              tag="draw_canvas", parent="canvas_border") as canvas:
                pass


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


def main(open_drawing_window):
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

                dpg.add_button(label="draw popup", callback=lambda: open_drawing_window("png", "title", 300))

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

        # with dpg.drawlist(width=300, height=300):  # or you could use dpg.add_drawlist and set parents manually
        #     dpg.draw_line((10, 10), (100, 100), color=(255, 0, 0, 255), thickness=1)
        #     dpg.draw_text((0, 0), "Origin", color=(250, 250, 250, 255), size=15)
        #     dpg.draw_arrow((50, 70), (100, 65), color=(0, 200, 255), thickness=1, size=10)
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
    main(open_drawing_window)
