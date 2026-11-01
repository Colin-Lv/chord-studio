Option Explicit
Dim shell
Set shell = CreateObject("WScript.Shell")
' Use the existing project store explicitly, without global environment changes.
shell.Environment("Process")("CHORD_DESKTOP_DATA_DIR") = "D:\ChordStudio\data"
shell.CurrentDirectory = "C:\Users\33573\AppData\Local\Programs\Chord Studio"
shell.Run Chr(34) & "C:\Users\33573\AppData\Local\Programs\Chord Studio\chord-studio.exe" & Chr(34), 1, False
