"""Turn recognised sounds into real keyboard and mouse actions.

Uses pynput, which works on macOS, Windows and Linux. On a Mac, the first time you
run control mode you must allow Terminal (or VS Code) in:
    System Settings -> Privacy & Security -> Accessibility
Otherwise the key presses are silently blocked by macOS.
"""

ACTIONS = {
    "next": "Right arrow (next slide, next photo, skip forward)",
    "previous": "Left arrow (previous slide, back)",
    "select": "Space (play/pause, select, page down)",
    "enter": "Enter (open, confirm)",
    "scroll_down": "Scroll down",
    "scroll_up": "Scroll up",
    "click": "Left mouse click where the pointer is",
    "toggle_listening": "Pause or resume listening",
    "none": "Do nothing",
}


class ActionController:
    """Performs actions. dry_run=True only prints them (for testing, or no permissions)."""

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.kb = self.mouse = None
        if not dry_run:
            try:
                from pynput import keyboard, mouse
                self._keyboard, self._mouse = keyboard, mouse
                self.kb = keyboard.Controller()
                self.mouse = mouse.Controller()
            except Exception as e:     # no display, missing permission or pynput not installed
                print(f"  ! Can't control the keyboard/mouse ({e}). Running in dry-run mode: "
                      "actions are only printed.")
                self.dry_run = True

    def perform(self, action):
        if action in ("none", "toggle_listening") or action not in ACTIONS:
            return
        if self.dry_run:
            return
        K = self._keyboard.Key
        keys = {"next": K.right, "previous": K.left, "select": K.space, "enter": K.enter}
        if action in keys:
            self.kb.press(keys[action])
            self.kb.release(keys[action])
        elif action == "scroll_down":
            self.mouse.scroll(0, -3)
        elif action == "scroll_up":
            self.mouse.scroll(0, 3)
        elif action == "click":
            self.mouse.click(self._mouse.Button.left)
