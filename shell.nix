{
  pkgs,
  inputs,
  ...
}:
pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    ffmpeg
    python313
    python313Packages.pip
    python313Packages.virtualenv

    python313Packages.fastapi
    python313Packages.pydantic
    python313Packages.python-multipart
    python313Packages.redis
    python313Packages.uvicorn
    python313Packages.python-ffmpeg
    python313Packages.celery
    python313Packages.requests

    python313Packages.noisereduce
    python313Packages.soundfile

    python313Packages.torch-bin
    python313Packages.torchaudio
    python313Packages.transformers
    python313Packages.tokenizers
  ];

  shellHook = ''
    echo "python dev environment"
    virtualenv venv
    source venv/bin/activate
  '';
}
