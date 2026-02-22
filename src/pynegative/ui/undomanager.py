import time
from typing import Any


class UndoManager:
    """Manages undo/redo state with batched changes."""

    def __init__(self, max_history: int = 50, batch_window: float = 1.0):
        """Initialize undo manager.

        Args:
            max_history: Maximum number of states to keep in history
            batch_window: Time window in seconds for batching changes
        """
        self._history = []
        self._current_index = -1
        self._max_history = max_history
        self._batch_window = batch_window
        self._last_push_time = 0
        self._last_description = ""

    def push_state(
        self, description: str, settings: dict[str, Any], rating: int
    ) -> bool:
        """Push a new state to the history.

        Args:
            description: Human-readable description of the change
            settings: Dictionary of all slider values
            rating: Current star rating

        Returns:
            True if state was added, False if batched with previous
        """
        current_time = time.time()

        # Check if we should batch with the previous state
        if (
            self._current_index >= 0
            and (current_time - self._last_push_time) < self._batch_window
            and self._last_description == description
            and description.startswith("Adjust")
        ):
            # Update the existing state instead of creating new one
            self._history[self._current_index] = {
                "timestamp": current_time,
                "description": description,
                "settings": settings.copy(),
                "rating": rating,
            }
            self._last_push_time = current_time
            return False

        # Remove any redo states (items after current index)
        if self._current_index < len(self._history) - 1:
            self._history = self._history[: self._current_index + 1]

        # Create new state
        state = {
            "timestamp": current_time,
            "description": description,
            "settings": settings.copy(),
            "rating": rating,
        }

        self._history.append(state)
        self._current_index += 1

        # Trim history if it exceeds max
        if len(self._history) > self._max_history:
            self._history.pop(0)
            self._current_index -= 1

        self._last_push_time = current_time
        self._last_description = description

        return True

    def undo(self) -> dict[str, Any] | None:
        """Undo the last action and return the previous state.

        Returns:
            The state to restore, or None if cannot undo
        """
        if self._current_index <= 0:
            return None

        self._current_index -= 1
        return self._history[self._current_index].copy()

    def redo(self) -> dict[str, Any] | None:
        """Redo the last undone action and return the state.

        Returns:
            The state to restore, or None if cannot redo
        """
        if self._current_index >= len(self._history) - 1:
            return None

        self._current_index += 1
        return self._history[self._current_index].copy()

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self._current_index > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self._current_index < len(self._history) - 1

    def clear(self):
        """Clear all history."""
        self._history = []
        self._current_index = -1
        self._last_push_time = 0
        self._last_description = ""

    def get_current_description(self) -> str:
        """Get description of current state."""
        if 0 <= self._current_index < len(self._history):
            return self._history[self._current_index]["description"]
        return ""

    def get_undo_description(self) -> str:
        """Get description of state that would be undone to."""
        if self._current_index > 0:
            return self._history[self._current_index - 1]["description"]
        return ""

    def get_redo_description(self) -> str:
        """Get description of state that would be redone."""
        if self._current_index < len(self._history) - 1:
            return self._history[self._current_index + 1]["description"]
        return ""
