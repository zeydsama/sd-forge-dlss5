def preload(parser):
    parser.add_argument(
        "--dlss5-runtime-dir",
        type=str,
        default=None,
        help="Custom path to DLSS 5 D3D12 worker runtime directory containing nvngx.dll, renodx-dlss5.addon64, etc.",
    )
    parser.add_argument(
        "--dlss5-debug",
        action="store_true",
        help="Enable verbose debugging logs for DLSS 5 Protocol v4 IPC communication.",
    )
