"""
Regression tests for GUI/loading.py fixes

Tests the critical changes we made to prevent the generation hang/crash:
1. ThreadedGenerator.map attribute initialization (was causing AttributeError)
2. Exception handling in ThreadedGenerator.run() (was silently failing)
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from queue import Queue

from DreamAtlas.GUI.loading import ThreadedGenerator, GeneratorLoadingWidget
from DreamAtlas.classes import DreamAtlasSettings, DominionsMap


class TestThreadedGeneratorInitialization:
    """Verify ThreadedGenerator properly initializes the map attribute"""

    def test_map_attribute_exists_after_init(self):
        """Regression: ThreadedGenerator.map was None before init, causing AttributeError"""
        queue = Queue()
        mock_ui = Mock()
        settings = DreamAtlasSettings(index=0)

        generator = ThreadedGenerator(queue, mock_ui, settings)

        # This is the critical fix - map must exist (even if None)
        assert hasattr(generator, 'map'), "ThreadedGenerator must have 'map' attribute"
        assert generator.map is None, "ThreadedGenerator.map should initialize as None"

    def test_map_attribute_type_consistency(self):
        """Verify map attribute can hold DominionsMap objects"""
        queue = Queue()
        mock_ui = Mock()
        settings = DreamAtlasSettings(index=0)

        generator = ThreadedGenerator(queue, mock_ui, settings)
        map_obj = DominionsMap()

        # Should be able to assign a map
        generator.map = map_obj
        assert generator.map is map_obj


class TestThreadedGeneratorExceptionHandling:
    """Verify ThreadedGenerator properly catches and reports exceptions"""

    def test_exception_in_generator_is_caught(self):
        """Regression: Exceptions in run() were silently failing, causing hung UI"""
        queue = Queue()
        mock_ui = Mock()
        settings = DreamAtlasSettings(index=0)

        generator = ThreadedGenerator(queue, mock_ui, settings)

        # Mock the generator_dreamatlas to raise an exception
        with patch('DreamAtlas.GUI.loading.generator_dreamatlas') as mock_gen:
            mock_gen.side_effect = ValueError("Test exception")

            # This should not raise - exception should be caught
            generator.run()

            # Queue should still have completion message
            assert not queue.empty(), "Queue should have completion message even after exception"
            msg = queue.get()
            assert msg == "Task finished", "Should signal completion even on error"

    def test_exception_types_are_caught(self):
        """Verify various exception types are handled"""
        queue = Queue()
        mock_ui = Mock()
        settings = DreamAtlasSettings(index=0)

        exception_types = [
            ValueError("Value error"),
            TypeError("Type error"),
            RuntimeError("Runtime error"),
            Exception("Generic exception"),
        ]

        for exc in exception_types:
            queue = Queue()
            generator = ThreadedGenerator(queue, mock_ui, settings)

            with patch('DreamAtlas.GUI.loading.generator_dreamatlas') as mock_gen:
                mock_gen.side_effect = exc

                # Should not raise
                generator.run()

                # Should still signal completion
                assert not queue.empty()


class TestGeneratorLoadingWidgetIntegration:
    """Test GeneratorLoadingWidget's use of ThreadedGenerator"""

    def test_generator_attribute_accessible(self):
        """Verify the fix that allows accessing generator.map in process_queue"""
        queue = Queue()
        mock_ui = Mock()
        settings = DreamAtlasSettings(index=0)

        generator = ThreadedGenerator(queue, mock_ui, settings)

        # This is what process_queue() tries to do:
        # self.master.map = self.generator.map
        # This should not raise AttributeError
        try:
            map_value = generator.map
            assert map_value is None  # Initially None
        except AttributeError:
            pytest.fail("ThreadedGenerator.map should be accessible")


class TestQueueCommunication:
    """Test thread-safe queue communication"""

    def test_task_finished_message_sent_on_success(self):
        """Verify completion message is sent"""
        queue = Queue()
        mock_ui = Mock()
        settings = DreamAtlasSettings(index=0)

        generator = ThreadedGenerator(queue, mock_ui, settings)

        with patch('DreamAtlas.GUI.loading.generator_dreamatlas') as mock_gen:
            test_map = DominionsMap()
            mock_gen.return_value = test_map

            generator.run()

            # Check queue has completion message
            assert not queue.empty()
            msg = queue.get()
            assert msg == "Task finished"

    def test_task_finished_message_sent_on_exception(self):
        """Verify completion message is sent even when generator fails"""
        queue = Queue()
        mock_ui = Mock()
        settings = DreamAtlasSettings(index=0)

        generator = ThreadedGenerator(queue, mock_ui, settings)

        with patch('DreamAtlas.GUI.loading.generator_dreamatlas') as mock_gen:
            mock_gen.side_effect = RuntimeError("Generation failed")

            generator.run()

            # Check queue still has completion message
            assert not queue.empty()
            msg = queue.get()
            assert msg == "Task finished"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
