from pynegative.ui.undomanager import UndoManager


def test_undo_manager_basic():
    manager = UndoManager()
    settings = {"exposure": 1.0}
    manager.push_state("Adjust Exposure", settings, 5)

    assert (
        manager.can_undo() is False
    )  # First state is base state, can't undo from it yet?
    # Actually, in most systems the first push is the current state.
    # Looking at the code: self._current_index starts at -1. push_state makes it 0.
    # undo() checks self._current_index <= 0.

    assert manager._current_index == 0
    assert manager.can_undo() is False

    manager.push_state("Adjust Blur", {"exposure": 1.0, "blur": 0.5}, 5)
    assert manager.can_undo() is True
    assert manager.get_current_description() == "Adjust Blur"
    assert manager.get_undo_description() == "Adjust Exposure"

    state = manager.undo()
    assert state["description"] == "Adjust Exposure"
    assert manager.can_undo() is False
    assert manager.can_redo() is True
    assert manager.get_redo_description() == "Adjust Blur"

    state = manager.redo()
    assert state["description"] == "Adjust Blur"


def test_undo_manager_batching():
    manager = UndoManager(batch_window=10.0)
    settings1 = {"exposure": 1.1}
    manager.push_state("Adjust Exposure", settings1, 4)

    settings2 = {"exposure": 1.2}
    # Should be batched
    pushed = manager.push_state("Adjust Exposure", settings2, 4)
    assert pushed is False
    assert len(manager._history) == 1
    assert manager._history[0]["settings"]["exposure"] == 1.2


def test_undo_manager_trimming():
    manager = UndoManager(max_history=3)
    manager.push_state("1", {}, 0)
    manager.push_state("2", {}, 0)
    manager.push_state("3", {}, 0)
    manager.push_state("4", {}, 0)

    assert len(manager._history) == 3
    assert manager._history[0]["description"] == "2"


def test_undo_manager_clear():
    manager = UndoManager()
    manager.push_state("1", {}, 0)
    manager.clear()
    assert len(manager._history) == 0
    assert manager._current_index == -1
