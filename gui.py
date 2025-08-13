from src.data.import_dataset import load_dataset, load_camera
from src.config.train.config import load_config
from src.spe.visualize import VisualizePose
from src.spe.spe_utils import SPEUtils
from src.modeling.model import import_model
from src.temporal.inference import Inference

import numpy as np
import os
import time
import torch
from collections import deque
import PIL.Image as PILImage

os.environ["KIVY_NO_CONSOLELOG"] = "1"

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.checkbox import CheckBox
from kivy.uix.spinner import Spinner, SpinnerOption
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time*1000:.3f} ms")
        return result
    return wrapper


def import_model_data(model_path, dataset_path):

    cfg = load_config(os.path.join(model_path, 'config.yaml'))
    camera = load_camera(dataset_path)
    spe_utils = SPEUtils(
        camera, cfg.MODEL.HEAD.ORI, cfg.MODEL.HEAD.N_ORI_BINS_PER_DIM, cfg.DATA.ORI_SMOOTH_FACTOR,
        cfg.MODEL.HEAD.ORI_DELETE_UNUSED_BINS, cfg.MODEL.HEAD.POS, cfg.MODEL.HEAD.N_POS_BINS_PER_DIM,
        cfg.DATA.POS_SMOOTH_FACTOR, cfg.MODEL.HEAD.KEYPOINTS_PATH
    )

    rot_augment = False
    other_augment = False
    data_shuffle = False
    batch_size = 1
    seed = 1001

    data, _ = load_dataset(spe_utils, dataset_path, batch_size, cfg.DATA.IMG_SIZE,
                           rot_augment, other_augment, data_shuffle, seed)
    splits = list(data.keys())  # Guarantee specific order for the splits if python 3.7 or above

    params_path = os.path.join(model_path, 'model', 'parameters.pt')
    bw_path = os.path.join(model_path, 'model', 'bit_width.json')
    bit_width_path = bw_path if os.path.exists(bw_path) else None

    model, bit_width = import_model(
        data, cfg.MODEL.BACKBONE.NAME, cfg.MODEL.HEAD.NAME, params_path, bit_width_path,
        manual_copy=False, residual=cfg.MODEL.BACKBONE.RESIDUAL, quantization=cfg.MODEL.QUANTIZATION,
        ori_mode=cfg.MODEL.HEAD.ORI, n_ori_bins=spe_utils.orientation.n_bins,
        pos_mode=cfg.MODEL.HEAD.POS, n_pos_bins=spe_utils.position.n_bins,
    )
    model.eval()

    # example image to convert the nodel to TS
    img, _ = next(iter(data[splits[0]]))
    model = torch.jit.trace(model, img['torch'])

    return model, data, splits, camera, spe_utils


class CustomSpinnerOption(SpinnerOption):
    def __init__(self, fontsize=30, **kwargs):
        super(CustomSpinnerOption, self).__init__(**kwargs)
        self.font_size = fontsize


class ColoredCheckBox(CheckBox):
    def __init__(self, color=(0, 0, 0, 1), **kwargs):
        super(ColoredCheckBox, self).__init__(**kwargs)
        self.color = color


class SpacecraftPoseGui(App):

    def __init__(self, **kwargs):
        super(SpacecraftPoseGui, self).__init__(**kwargs)

        # Define datasets, models and devices to execute the inference
        self.dataset_list = {
            'speed': "../datasets/speed",
            'speed_plus': "../datasets/speed_plus",
            'dspeed': "../datasets/dspeed/still",
            'dspeed_video': "../datasets/dspeed/video",
        }

        self.model_list = {
            'murso_speed': "models/murso_fp32_speed",
            'murso_dspeed': "models/murso_fp32_dspeed",
            'mursop_speed': "models/mursop_fp32_speed",
            'mursop_dspeed': "models/mursop_fp32_dspeed",
            'mursop_dspeed_large': "models/mursop_fp32_dspeed_large_image",
        }

        self.device_list = [
            'gpu_host',
            'cpu_host',
            'gpu_jetson',
        ]

        # Current model, dataset and inference device selected in the GUI
        self.model = list(self.model_list.keys())[0]
        self.dataset = list(self.dataset_list.keys())[0]
        self.device = self.device_list[0]

        model, self.data, self.split_list, camera, self.spe_utils = import_model_data(
            self.model_list[self.model],
            self.dataset_list[self.dataset]
        )

        self.split = self.split_list[0]
        self.data_iterator = iter(self.data[self.split])

        self.visualization = VisualizePose(self.spe_utils)
        self.spe_inference = Inference(model, self.device, self.spe_utils)

        # Spacecraft Pose Estimation variables
        self.image_index = 0  # Image index in the dataset
        self.image = None
        self.true_pose = None
        self.pred_pose = None
        self.pose_temporal = None
        self.metrics = None
        self.metrics_temp = None  # Temporal
        self.inference_latency_ms = None

        # Control variables
        self.is_running = False
        self.show_true_pose = False
        self.show_pred_pose = False
        self.show_temp_pose = False
        self.show_true_bbox = False
        self.show_pred_bbox = False
        self.show_temp_bbox = False
        self.show_true_keypoints = False
        self.show_pred_keypoints = False
        self.show_temp_keypoints = False

        # Define GUI attributes before the build method
        self.bg_rect = None
        self.image_display = None
        self.image_name_label = None
        self.split_spinner = None
        self.run_stop_button = None
        self.metrics_table = None
        self.target_distance_label = None
        self.inference_latency_label = None

        # Periodic event created for video sequences
        self.run_event = None

        # Texture to print the image in the GUI
        self.texture = None

        self.image_run = deque()
        self.true_pose_run = deque()
        self.pred_still_pose_run = deque()
        self.pred_video_pose_run = deque()
        self.metrics_still_run = deque()
        self.metrics_video_run = deque()
        self.inference_latency_ms_run = deque()

    def build(self):
        fontsize = 40
        padding = 20
        spacing_global = 60
        spacing_local = 20
        cb_color = (0, 0, 0, 1)

        Window.size = (1920, 1200)  # Set the window size to 1920x1200
        root = BoxLayout(orientation='horizontal', padding=padding)

        # Define GUI background color
        with root.canvas.before:
            Color(109/255, 126/255, 143/255, 1)  # Set background color
            self.bg_rect = Rectangle(size=root.size, pos=root.pos)
        root.bind(size=self.update_bg_rect, pos=self.update_bg_rect)

        # LEFT PANEL: Image display and name
        left_panel = BoxLayout(orientation='vertical', size_hint=(0.7, 1), padding=padding)
        self.image_display = Image(size_hint=(1, 0.9), allow_stretch=True)
        self.image_name_label = Label(text="image_name.png", font_size=fontsize, size_hint=(1, 0.1))
        left_panel.add_widget(self.image_name_label)
        left_panel.add_widget(self.image_display)

        # RIGHT PANEL: Controls and infos
        right_panel = BoxLayout(orientation='vertical', size_hint=(0.3, 1), spacing=spacing_global, padding=padding)

        # ROW 1: Model and Device selectors
        model_device_layout = BoxLayout(orientation='horizontal', spacing=spacing_local)
        model_spinner = Spinner(
            text=self.model, values=list(self.model_list.keys()),
            font_size=fontsize,
            option_cls=lambda **kwargs: CustomSpinnerOption(fontsize=fontsize, **kwargs),
        )
        model_spinner.bind(text=self.on_model_spinner_select)
        device_spinner = Spinner(
            text=self.device, values=self.device_list,
            font_size=fontsize,
            option_cls=lambda **kwargs: CustomSpinnerOption(fontsize=fontsize, **kwargs),
        )
        device_spinner.bind(text=self.on_device_spinner_select)
        model_device_layout.add_widget(model_spinner)
        model_device_layout.add_widget(device_spinner)
        right_panel.add_widget(model_device_layout)

        # ROW 2: Dataset and Split selectors
        dataset_split_layout = BoxLayout(orientation='horizontal', spacing=spacing_local)
        dataset_spinner = Spinner(
            text=self.dataset, values=list(self.dataset_list.keys()),
            font_size=fontsize,
            option_cls=lambda **kwargs: CustomSpinnerOption(fontsize=fontsize, **kwargs)
        )
        dataset_spinner.bind(text=self.on_dataset_spinner_select)
        self.split_spinner = Spinner(
            text=self.split, values=self.split_list,
            font_size=fontsize,
            option_cls=lambda **kwargs: CustomSpinnerOption(fontsize=fontsize, **kwargs)
        )
        self.split_spinner.bind(text=self.on_split_spinner_select)
        dataset_split_layout.add_widget(dataset_spinner)
        dataset_split_layout.add_widget(self.split_spinner)
        right_panel.add_widget(dataset_split_layout)

        # ROW 3: Navigation buttons
        nav_layout = BoxLayout(orientation='horizontal', spacing=spacing_local)
        prev_button = Button(text='Previous image', font_size=fontsize)
        prev_button.bind(on_press=self.on_prev_button_press)
        next_button = Button(text='Next image', font_size=fontsize)
        next_button.bind(on_press=self.on_next_button_press)
        nav_layout.add_widget(prev_button)
        nav_layout.add_widget(next_button)
        right_panel.add_widget(nav_layout)

        # ROW 4: Run and Reset buttons
        run_reset_layout = BoxLayout(orientation='horizontal', spacing=spacing_local)
        self.run_stop_button = Button(text='Run', font_size=fontsize)
        self.run_stop_button.bind(on_press=self.on_run_stop_button_press)
        reset_button = Button(text='Reset', font_size=fontsize)
        reset_button.bind(on_press=self.on_reset_button_press)
        run_reset_layout.add_widget(self.run_stop_button)
        run_reset_layout.add_widget(reset_button)
        right_panel.add_widget(run_reset_layout)

        # TABLE 1: Visualization of the pose
        # Mapping checkbox types to their respective callback methods
        checkbox_callbacks = {
            'true_pose': self.on_true_pose_checkbox_active,
            'pred_pose': self.on_pred_pose_checkbox_active,
            'temp_pose': self.on_temp_pose_checkbox_active,
            'true_bbox': self.on_true_bbox_checkbox_active,
            'pred_bbox': self.on_pred_bbox_checkbox_active,
            'temp_bbox': self.on_temp_bbox_checkbox_active,
            'true_keypoints': self.on_true_keypoints_checkbox_active,
            'pred_keypoints': self.on_pred_keypoints_checkbox_active,
            'temp_keypoints': self.on_temp_keypoints_checkbox_active
        }

        # Visualization labels and checkbox identifiers
        visu_labels = [
            ['Visualization', 'True', 'Still', 'Temporal'],
            ['Arrows', 'true_pose', 'pred_pose', 'temp_pose'],
            ['Bounding box', 'true_bbox', 'pred_bbox', 'temp_bbox'],
            ['Keypoints', 'true_keypoints', 'pred_keypoints', 'temp_keypoints']
        ]

        # Create the layout
        pose_visu_layout = GridLayout(cols=4, rows=4)

        # Iterate through the visualization structure
        for row in visu_labels:
            for item in row:
                if item in checkbox_callbacks:
                    # Create and bind checkboxes
                    checkbox = ColoredCheckBox(color=cb_color)
                    checkbox.bind(active=checkbox_callbacks[item])
                    pose_visu_layout.add_widget(checkbox)
                else:
                    # Add labels
                    pose_visu_layout.add_widget(Label(text=item, font_size=fontsize))

        # Add the layout to the right panel
        right_panel.add_widget(pose_visu_layout)

        # TABLE 2: Pose metrics
        self.metrics_table = GridLayout(cols=3, rows=4)
        headings = ["Metric", "still", "temporal"]
        self.metrics_table.add_widget(Label(text=headings[0], font_size=fontsize))
        self.metrics_table.add_widget(Label(text=headings[1], font_size=fontsize))
        self.metrics_table.add_widget(Label(text=headings[2], font_size=fontsize))
        metrics_val = [
            ["POSE error", "X", "X"],
            ["ORI error (°)", "X", "X"],
            ["POS error (m)", "X", "X"]
        ]
        for row in metrics_val:
            for col in row:
                self.metrics_table.add_widget(Label(text=col, font_size=fontsize))
        right_panel.add_widget(self.metrics_table)

        # TABLE 3: Additional info
        info_layout = GridLayout(cols=1, rows=2)
        self.target_distance_label = Label(text="Target distance = X m", font_size=fontsize)
        self.inference_latency_label = Label(text="Latency = X ms", font_size=fontsize)
        info_layout.add_widget(self.target_distance_label)
        info_layout.add_widget(self.inference_latency_label)
        right_panel.add_widget(info_layout)

        root.add_widget(left_panel)
        root.add_widget(right_panel)

        # Print first image
        self.load_image()
        self.inference()
        self.update_gui()

        return root

    def update_bg_rect(self, instance, value):
        self.bg_rect.size = instance.size
        self.bg_rect.pos = instance.pos

    # @measure_execution_time
    def load_image(self):
        if self.is_running:
            try:
                self.image, self.true_pose = next(self.data_iterator)
            except StopIteration:
                # go back to image zero (reset iterator)
                self.data_iterator = iter(self.data[self.split])
                self.image, self.true_pose = next(self.data_iterator)
                self.spe_inference.reset()

            # Remove batch dimension from data
            self.image = {key: value[0] for key, value in self.image.items()}
            self.true_pose = {key: value[0].numpy() for key, value in self.true_pose.items()}
        else:
            self.image, self.true_pose = self.data[self.split].dataset[self.image_index]
            self.true_pose = {key: value.numpy() for key, value in self.true_pose.items()}

    # @measure_execution_time
    def inference(self):

        temporal_type = "Adaptative" if "video" in self.dataset else None
        self.pred_pose, self.inference_latency_ms, self.pose_temporal = self.spe_inference.predict(
            self.image['torch'].unsqueeze(0), temporal_type
        )

        # DEBUG: visualize error introduced by soft-classification
        # self.pred_pose['ori_soft'] = self.spe_utils.orientation.encode(self.true_pose['ori'])
        # self.pred_pose['ori'], _ = self.spe_utils.orientation.decode(self.pred_pose['ori_soft'])
        # self.pred_pose['pos_soft'] = self.spe_utils.position.encode(self.true_pose['pos'])
        # self.pred_pose['pos'] = self.spe_utils.position.decode(self.pred_pose['pos_soft'])
        # self.pred_pose['keypoints'] = self.spe_utils.keypoints.create_keypoints2d(
        #     self.pred_pose['ori'], self.pred_pose['pos'])
        # self.pred_pose['bbox'] = self.spe_utils.keypoints.create_bbox_from_keypoints(
        #     self.pred_pose['keypoints']
        # )

        # Get evaluation metrics
        self.metrics = self.spe_utils.get_score(
            {key:np.expand_dims(value, axis=0) for key,value in self.true_pose.items()},
            {key:np.expand_dims(value, axis=0) for key,value in self.pred_pose.items()},
        )
        # self.metrics = {key: np.squeeze(value, axis=0) for key, value in metrics.items()} # remove batch dimension
        if self.pose_temporal is not None:
            # print(f"temp pose:\n{self.pose_temporal}\n")
            self.metrics_temp = self.spe_utils.get_score(
                {key: np.expand_dims(value, axis=0) for key, value in self.true_pose.items()},
                {key: np.expand_dims(value, axis=0) for key, value in self.pose_temporal.items()},
            )
            # self.metrics_temp = {key: np.squeeze(value, axis=0) for key, value in metrics_temp.items()}  # remove batch dimension
        else:
            self.metrics_temp = None

    # @measure_execution_time
    def numpy_to_texture(self, np_image: np.ndarray):
        # Ensure the numpy array is in the correct format (e.g., uint8)
        np_image = np_image.astype(np.uint8)
        height, width, _ = np_image.shape

        # Create the first time, else reuse it as image size does not change
        if self.texture is None:
            self.texture = Texture.create(size=(width, height), colorfmt='rgb')

        # Blit the buffer into the texture
        self.texture.blit_buffer(np_image.tobytes(), colorfmt='rgb', bufferfmt='ubyte')

        # Flip the image vertically
        # In Kivy, textures and images are typically rendered with the origin (0, 0) at the bottom left,
        # while in NumPy the origin is at the top left.
        self.texture.flip_vertical()

        return self.texture

    # @measure_execution_time
    def update_gui(self):
        # Update image
        image_path = self.image['path']

        # Generate visuals for articles/thesis
        # if image_path == "../datasets/dspeed/video/TIR_eb/images/00863.png":
        #     image = self.visualization.add_visualization(
        #         self.image['original'].numpy(), self.true_pose, self.pred_pose, self.pose_temporal,
        #         False, False, False,
        #         False, False, False,
        #         True, False, False
        #     )
        #
        #     img = PILImage.fromarray(image)
        #     img.save("output.png")

        image = self.visualization.add_visualization(
            self.image['original'].numpy(), self.true_pose, self.pred_pose, self.pose_temporal,
            self.show_true_pose, self.show_pred_pose, self.show_temp_pose,
            self.show_true_bbox, self.show_pred_bbox, self.show_temp_bbox,
            self.show_true_keypoints, self.show_pred_keypoints, self.show_temp_keypoints
        )

        texture = self.numpy_to_texture(image)
        self.image_display.texture = texture
        self.image_name_label.text = os.path.basename(image_path)

        # Update metrics & infos

        # Metrics table
        self.metrics_table.children[7].text = f"{self.metrics['esa_score']:.3f}"
        self.metrics_table.children[4].text = f"{self.metrics['ori_error']:.3f}"
        self.metrics_table.children[1].text = f"{self.metrics['pos_error']:.3f}"
        if self.metrics_temp is not None:
            self.metrics_table.children[6].text = f"{self.metrics_temp['esa_score']:.3f}"
            self.metrics_table.children[3].text = f"{self.metrics_temp['ori_error']:.3f}"
            self.metrics_table.children[0].text = f"{self.metrics_temp['pos_error']:.3f}"
        else:
            self.metrics_table.children[6].text = "X"
            self.metrics_table.children[3].text = "X"
            self.metrics_table.children[0].text = "X"

        # Additional info
        self.target_distance_label.text = f"Target distance = {np.linalg.norm(self.true_pose['pos']):.1f} m"
        self.inference_latency_label.text = f"Latency = {self.inference_latency_ms:.1f} ms"

    def on_model_spinner_select(self, spinner, text):
        self.model = text
        model, self.data, splits, camera, self.spe_utils = import_model_data(
            self.model_list[self.model],
            self.dataset_list[self.dataset]
        )
        self.spe_inference.update(model, self.spe_utils)

    def on_device_spinner_select(self, spinner, text):
        self.device = text
        # print(self.model)
        self.spe_inference.select_inference_engine(self.device, self.model)
        # self.spe_inference.select_inference_engine(self.device)

    def on_dataset_spinner_select(self, spinner, text):
        self.dataset = text
        model, self.data, self.split_list, camera, self.spe_utils = import_model_data(
            self.model_list[self.model],
            self.dataset_list[self.dataset]
        )
        self.image_index = 0
        self.split = self.split_list[0]
        self.split_spinner.values = self.split_list
        self.split_spinner.text = self.split
        self.data_iterator = iter(self.data[self.split])
        self.spe_inference.reset()
        self.load_image()
        self.inference()
        self.update_gui()

    def on_split_spinner_select(self, spinner, text):
        self.image_index = 0
        self.split = text
        self.data_iterator = iter(self.data[self.split])
        self.load_image()
        self.spe_inference.reset()
        self.inference()
        self.update_gui()

    def on_prev_button_press(self, instance):
        self.image_index -= 1
        self.image_index = self.image_index % len(self.data[self.split])
        if self.image_index == 0:
            self.data_iterator = iter(self.data[self.split])
        self.spe_inference.reset()  # Need to reset for temporal coherence on video data
        self.load_image()
        self.inference()
        self.update_gui()

    def on_next_button_press(self, instance):
        self.image_index += 1
        self.image_index = self.image_index % len(self.data[self.split])
        if self.image_index == 0:
            self.data_iterator = iter(self.data[self.split])
            self.spe_inference.reset()
        self.load_image()
        self.inference()
        self.update_gui()

    def on_run_stop_button_press(self, instance):
        self.is_running = not self.is_running
        self.run_stop_button.text = 'Stop' if self.is_running else 'Run'

        if self.is_running:
            # Schedule the continuous processing if checkbox is active
            # self.run_event = Clock.schedule_interval(self.run_continuous_processing, 1.0 / 25.0)  # 25 FPS
            self.run_event = Clock.schedule_once(self.run_continuous_processing, 0)
        else:
            # Cancel the scheduled event if checkbox is inactive
            if self.run_event:
                self.run_event.cancel()
                self.run_event = None

    def run_continuous_processing(self, dt):
        """
        dt = time between two consecutive calls to this function
        """
        # (f'dt = {dt*1000:.1f} ms')
        self.image_index = (self.image_index + 1) % len(self.data[self.split])
        self.load_image()
        self.inference()
        self.update_gui()

        # Schedule the next call to maintain exact intervals
        # remaining_time = max(0, interval - dt)
        # print(f'Next call in {remaining_time * 1000:.1f} ms')
        self.run_event = Clock.schedule_once(self.run_continuous_processing, 0)

    def on_reset_button_press(self, instance):
        self.data_iterator = iter(self.data[self.split])
        self.image_index = 0
        self.spe_inference.reset()
        self.load_image()
        self.inference()
        self.update_gui()

    def on_true_pose_checkbox_active(self, checkbox, value):
        if self.show_true_pose != value:
            print(f"Showing true pose: {value}")
            self.show_true_pose = value
            self.update_gui()

    def on_pred_pose_checkbox_active(self, checkbox, value):
        if self.show_pred_pose != value:
            self.show_pred_pose = value
            self.update_gui()

    def on_temp_pose_checkbox_active(self, checkbox, value):
        if self.show_temp_pose != value:
            self.show_temp_pose = value
            self.update_gui()

    def on_true_bbox_checkbox_active(self, checkbox, value):
        if self.show_true_bbox != value:
            self.show_true_bbox = value
            self.update_gui()

    def on_pred_bbox_checkbox_active(self, checkbox, value):
        if self.show_pred_bbox != value:
            self.show_pred_bbox = value
            self.update_gui()

    def on_temp_bbox_checkbox_active(self, checkbox, value):
        if self.show_temp_bbox != value:
            self.show_temp_bbox = value
            self.update_gui()

    def on_true_keypoints_checkbox_active(self, checkbox, value):
        if self.show_true_keypoints != value:
            self.show_true_keypoints = value
            self.update_gui()

    def on_pred_keypoints_checkbox_active(self, checkbox, value):
        if self.show_pred_keypoints != value:
            self.show_pred_keypoints = value
            self.update_gui()

    def on_temp_keypoints_checkbox_active(self, checkbox, value):
        if self.show_temp_keypoints != value:
            self.show_temp_keypoints = value
            self.update_gui()

    def on_stop(self):
        self.spe_inference.close_jetson()


if __name__ == '__main__':

    SpacecraftPoseGui().run()
