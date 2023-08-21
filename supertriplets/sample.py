import os
from typing import Dict


class TextSample:
    def __init__(self, text: str, label: int) -> None:
        self.text = text
        self.label = label
        self._validate_input()

    def _validate_input(self) -> None:
        assert type(self.text) == str, f"{self.__class__.__name__} 'text' should be a string"
        assert type(self.label) == int, f"{self.__class__.__name__} 'label' should be an integer"

    def data(self) -> Dict:
        return dict(text=self.text, label=self.label)

    def __repr__(self) -> str:
        return f"text={self.text}, label={self.label}"


class ImageSample:
    def __init__(self, image_path: str, label: int) -> None:
        self.image_path = image_path
        self.label = label
        self._validate_input()

    def _validate_input(self) -> None:
        assert type(self.image_path) == str, f"{self.__class__.__name__} 'image_path' should be a string"
        assert os.path.exists(self.image_path), f"{self.__class__.__name__} 'image_path' should exist"
        assert type(self.label) == int, f"{self.__class__.__name__} 'label' should be an integer"

    def data(self) -> Dict:
        return dict(image_path=self.image_path, label=self.label)

    def __repr__(self) -> str:
        return f"image_path={self.image_path}, label={self.label}"


class TextImageSample:
    def __init__(self, text: str, image_path: str, label: int) -> None:
        self.text = text
        self.image_path = image_path
        self.label = label
        self._validate_input()

    def _validate_input(self) -> None:
        assert type(self.text) == str, f"{self.__class__.__name__} 'text' should be a string"
        assert type(self.image_path) == str, f"{self.__class__.__name__} 'image_path' should be a string"
        assert os.path.exists(self.image_path), f"{self.__class__.__name__} 'image_path' should exist"
        assert type(self.label) == int, f"{self.__class__.__name__} 'label' should be an integer"

    def data(self) -> Dict:
        return dict(text=self.text, image_path=self.image_path, label=self.label)

    def __repr__(self) -> str:
        return f"text={self.text}, image_path={self.image_path}, label={self.label}"
