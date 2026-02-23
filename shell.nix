{
  pkgs,
  inputs,
  ...
}:
pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    ffmpeg
    python314
    python314Packages.pip
    python314Packages.virtualenv

    python314Packages.fastapi
    python314Packages.pydantic
    python314Packages.python-multipart
    python314Packages.redis
    python314Packages.uvicorn
  ];

  shellHook = ''
    echo "python dev environment"
    virtualenv venv
    source venv/bin/activate
  '';
}
