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
    # python314Packages.torch
    # python314Packages.torchWithCuda
    python314Packages.torch-bin
    python314Packages.gradio
    python314Packages.transformers
    python314Packages.tokenizers
    python314Packages.scikit-learn
    python314Packages.librosa
    python314Packages.flask

    # noise clean
    python314Packages.noisereduce
    python314Packages.soundfile

    # GigaAM-v3 failed
    python314Packages.hydra-core
    python314Packages.torchcodec
    python314Packages.sentencepiece
    python314Packages.torchaudio
    # python313Packages.pyannote-audio

    # hugginface
    inputs.huggin.legacyPackages.x86_64-linux.python314Packages.huggingface-hub
  ];

  shellHook = ''
    echo "python dev environment"
    virtualenv venv
    source venv/bin/activate
  '';
}
