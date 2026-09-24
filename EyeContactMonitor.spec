# -*- mode: python ; coding: utf-8 -*-

import sys
import os

block_cipher = None

a = Analysis(
    ['app/live_eye_contactv5.py'],
    pathex=['.'],
    binaries=[],
    datas=[('models/best_gaze_model.pt', 'models')],
    hiddenimports=[
        'mediapipe',
        'win32gui',
        'win32con',
        'winsound',
        'torch',
        'cv2'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.cipher, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='EyeContactMonitor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='EyeContactMonitor',
)
