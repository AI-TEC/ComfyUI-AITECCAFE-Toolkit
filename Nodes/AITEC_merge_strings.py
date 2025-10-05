class CustomStringMergeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_string1": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "use_string2": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "use_string3": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
            },
            "optional": {  # string1～3をoptionalに移動
                "string1": ("STRING", {"default": ""}),
                "string2": ("STRING", {"default": ""}),
                "string3": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("merged_string",)
    
    FUNCTION = "merge_strings"
    
    CATEGORY = "text"
    
    def merge_strings(self, string1="", string2="", string3="", use_string1=True, use_string2=True, use_string3=False):
        print(f"Switches: use_string1={use_string1}, use_string2={use_string2}, use_string3={use_string3}")
        strings = []
        if use_string1 and string1:
            strings.append(string1)
        if use_string2 and string2:
            strings.append(string2)
        if use_string3 and string3:
            strings.append(string3)
        
        merged = " \n".join(strings) if strings else ""
        return (merged,)

NODE_CLASS_MAPPINGS = {
    "CustomStringMerge": CustomStringMergeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomStringMerge": "AITEC Custom String Merge"
}