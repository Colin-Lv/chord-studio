"""Check the small local port-selection/handshake correction without a real DAW."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fl_studio_mcp.utils.midi_connection import MIDIConnection


class ConnectionGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path_patch = patch("fl_studio_mcp.utils.midi_connection._get_fl_hardware_dir", return_value=Path(self.temp.name))
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        self.env_patch = patch.dict(os.environ, {"FL_MIDI_PORT": "FLStudioMCP"})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    @patch("mido.open_output")
    @patch("mido.get_output_names", return_value=["Microsoft GS Wavetable Synth 0"])
    def test_never_falls_back_to_system_synth(self, names, open_port):
        conn = MIDIConnection()
        self.assertFalse(conn.connect())
        open_port.assert_not_called()
        self.assertFalse(conn.is_connected)

    @patch("mido.open_output")
    @patch("mido.get_output_names", return_value=["FLStudioMCP 1", "FLStudioMCP 2"])
    def test_ambiguous_port_is_rejected(self, names, open_port):
        self.assertFalse(MIDIConnection().connect())
        open_port.assert_not_called()

    @patch("mido.open_output")
    @patch("mido.get_output_names", return_value=["FLStudioMCP 1"])
    def test_open_port_without_fl_response_is_not_connected(self, names, open_port):
        conn = MIDIConnection()
        with patch.object(conn, "send_command", return_value={"success": False, "error": "timeout"}):
            self.assertFalse(conn.connect())
        self.assertFalse(conn.is_connected)
        open_port.return_value.close.assert_called_once()

    @patch("mido.open_output")
    @patch("mido.get_output_names", return_value=["FLStudioMCP 1"])
    def test_valid_fl_response_confirms_connection(self, names, open_port):
        conn = MIDIConnection()
        with patch.object(conn, "send_command", return_value={"success": True, "is_playing": False}) as probe:
            self.assertTrue(conn.connect())
            probe.assert_called_once_with("transport.getStatus")
        self.assertTrue(conn.is_connected)
        conn.disconnect()


if __name__ == "__main__":
    unittest.main()
