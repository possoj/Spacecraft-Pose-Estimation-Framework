"""
Copyright (c) 2025 Julien Posso
"""

import json
from PIL import Image
import numpy as np
import os

from src.spe.spe_utils import SPEUtils
from src.spe.visualize import VisualizePose

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
import kivy.uix.image
from kivy.uix.checkbox import CheckBox
from kivy.uix.spinner import Spinner, SpinnerOption
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture


class CustomSpinnerOption(SpinnerOption):
    def __init__(self, fontsize=30, **kwargs):
        super(CustomSpinnerOption, self).__init__(**kwargs)
        self.font_size = fontsize


class ColoredCheckBox(CheckBox):
    def __init__(self, color=(0, 0, 0, 1), **kwargs):
        super(ColoredCheckBox, self).__init__(**kwargs)
        self.color = color


class SpacecraftPoseGui(App):
    """
    Graphical interface for visualizing spacecraft poses and trajectories using wireframe rendering.

    This GUI allows users to inspect still poses or complete orbital trajectories of a satellite
    using only annotation data (e.g., JSON files with 6-DoF poses). The spacecraft is rendered as
    a wireframe over a black background, without requiring any external rendering engine.

    Main features:
    - Displays pose annotations frame by frame using a wireframe model of the satellite.
    - Navigates through dataset splits (e.g., train, val, test).
    - Supports automatic playback of trajectories.
    - Displays pose-related metadata such as image name and target distance.
    - Provides an interactive Kivy-based GUI for exploration and validation.
    """

    def __init__(self, splits_path: str, spe_utils: SPEUtils, **kwargs):
        """
        Initializes the spacecraft pose visualization GUI.

        Args:
            splits_path (str): Path to the directory containing dataset splits (produced by create_dspeed.py)
            and pose.json files.
            spe_utils (SPEUtils): Utility object with camera parameters and pose helpers.
            **kwargs: Additional arguments passed to the Kivy App superclass.
        """
        super(SpacecraftPoseGui, self).__init__(**kwargs)

        self.splits_path = splits_path
        self.spe_utils = spe_utils

        self.splits = os.listdir(self.splits_path)
        self.split = self.splits[0]
        with open(os.path.join(splits_path, self.split, 'pose.json'), 'r') as f:
            self.pose_dict = json.load(f)

        self.image = np.array(Image.new('RGB', (spe_utils.camera.nu, spe_utils.camera.nv), (0, 0, 0)))

        self.visualization = VisualizePose(self.spe_utils)

        self.image_index = 0  # Image index in the dataset

        self.image_path, self.true_pose = list(self.pose_dict.items())[self.image_index]

        # Define GUI attributes before the build method
        self.bg_rect = None
        self.image_display = None
        self.image_name_label = None
        self.split_spinner = None
        self.run_stop_button = None
        self.target_distance_label = None

        # Periodic event created for video sequences
        self.run_event = None

        self.is_running = False

        # Texture to print the image in the GUI
        self.texture = None

    def build(self):
        fontsize = 40
        padding = 20
        spacing_global = 60
        spacing_local = 20

        Window.size = (1920, 1200)  # Set the window size to 1920x1200
        root = BoxLayout(orientation='horizontal', padding=padding)

        # Define GUI background color
        with root.canvas.before:
            Color(109 / 255, 126 / 255, 143 / 255, 1)  # Set background color
            self.bg_rect = Rectangle(size=root.size, pos=root.pos)
        root.bind(size=self.update_bg_rect, pos=self.update_bg_rect)

        # LEFT PANEL: Image display and name
        left_panel = BoxLayout(orientation='vertical', size_hint=(0.7, 1), padding=padding)
        self.image_display = kivy.uix.image.Image(size_hint=(1, 0.9), allow_stretch=True)
        self.image_name_label = Label(text="image_name.png", font_size=fontsize, size_hint=(1, 0.1))
        left_panel.add_widget(self.image_name_label)
        left_panel.add_widget(self.image_display)

        # RIGHT PANEL: Controls and infos
        right_panel = BoxLayout(orientation='vertical', size_hint=(0.3, 1), spacing=spacing_global, padding=padding)

        # ROW: Dataset and Split selectors
        dataset_split_layout = BoxLayout(orientation='horizontal', spacing=spacing_local)
        self.split_spinner = Spinner(
            text=self.split, values=self.splits,
            font_size=fontsize,
            option_cls=lambda **kwargs: CustomSpinnerOption(fontsize=fontsize, **kwargs)
        )
        self.split_spinner.bind(text=self.on_split_spinner_select)
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

        # TABLE 3: Additional info
        self.target_distance_label = Label(text="Target distance = X m", font_size=fontsize)
        right_panel.add_widget(self.target_distance_label)

        root.add_widget(left_panel)
        root.add_widget(right_panel)

        # Print first image
        self.update_gui()

        return root

    def update_bg_rect(self, instance, value):
        self.bg_rect.size = instance.size
        self.bg_rect.pos = instance.pos

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

    def update_gui(self):
        # Update image
        self.image_path, self.true_pose = list(self.pose_dict.items())[self.image_index]
        self.true_pose['keypoints'] = self.spe_utils.keypoints.create_keypoints2d(
            self.true_pose['ori'], self.true_pose['pos'])

        # map true pose to pred pose for the green/yellow color
        image = self.visualization.add_visualization(self.image, pred_pose=self.true_pose, show_pred_keypoints=True)

        texture = self.numpy_to_texture(image)
        self.image_display.texture = texture
        self.image_name_label.text = os.path.basename(self.image_path)

        # Additional info
        self.target_distance_label.text = f"Target distance = {np.linalg.norm(self.true_pose['pos']):.1f} m"

    def on_split_spinner_select(self, spinner, text):
        self.image_index = 0
        self.split = text
        with open(os.path.join(self.splits_path, self.split, 'pose.json'), 'r') as f:
            self.pose_dict = json.load(f)
        self.update_gui()

    def on_prev_button_press(self, instance):
        self.image_index -= 1
        self.image_index = self.image_index % len(self.pose_dict)
        self.update_gui()

    def on_next_button_press(self, instance):
        self.image_index += 1
        self.image_index = self.image_index % len(self.pose_dict)
        self.update_gui()

    def on_run_stop_button_press(self, instance):
        self.is_running = not self.is_running
        self.run_stop_button.text = 'Stop' if self.is_running else 'Run'

        if self.is_running:
            # Schedule the continuous processing if checkbox is active
            self.run_event = Clock.schedule_interval(self.run_continuous_processing, 1.0 / 25.0)  # 25 FPS
        else:
            # Cancel the scheduled event if checkbox is inactive
            if self.run_event:
                self.run_event.cancel()
                self.run_event = None

    def run_continuous_processing(self, dt):
        # print(f"dt = {dt*1000:.1f} ms")
        self.image_index += 1
        self.image_index = self.image_index % len(self.pose_dict)
        self.update_gui()

    def on_reset_button_press(self, instance):
        print(f'Reset')
        self.image_index = 0
        self.update_gui()
