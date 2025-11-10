"""
Example Nodes - Templates and demonstrations
"""


class ExampleLibiglNode:
    """
    Example node demonstrating the ComfyUI node structure.
    This is a cookie-cutter template - replace with actual functionality.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input parameters for this node.

        Returns:
            dict: Input type definitions with 'required' and 'optional' keys
        """
        return {
            "required": {
                "text_input": ("STRING", {
                    "default": "Hello from GeomPack!",
                    "multiline": False
                }),
                "number_input": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "float_input": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "mode": (["option1", "option2", "option3"], {
                    "default": "option1"
                }),
            },
            "optional": {
                "optional_input": ("STRING", {
                    "default": ""
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT")
    RETURN_NAMES = ("output_text", "output_number", "output_float")
    FUNCTION = "execute"
    CATEGORY = "geompack/examples"
    OUTPUT_NODE = False

    def execute(self, text_input, number_input, float_input, mode, optional_input=""):
        """
        Execute the node logic.

        Args:
            text_input: String input
            number_input: Integer input
            float_input: Float input
            mode: Selected mode from combo box
            optional_input: Optional string input

        Returns:
            tuple: (output_text, output_number, output_float)
        """
        # Example processing
        result_text = f"{text_input} | Mode: {mode}"
        if optional_input:
            result_text += f" | Optional: {optional_input}"

        result_number = number_input * 2
        result_float = float_input * 1.5

        print(f"[ExampleNode] Processing: {result_text}")

        return (result_text, result_number, result_float)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackExampleNode": ExampleLibiglNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackExampleNode": "Example Node",
}
