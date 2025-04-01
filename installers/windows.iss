[Setup]
AppName=openMotor
AppVersion=0.6.0
WizardStyle=modern
DefaultDirName={autopf}\openMotor
DefaultGroupName=openMotor
UninstallDisplayIcon={app}\openMotor.exe
Compression=lzma2
SolidCompression=yes

[Files]
Source: "../dist/openMotor/*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{group}\openMotor"; Filename: "{app}\openMotor.exe"

