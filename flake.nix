{
  description = "CueBridge development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python313;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            git
            gitleaks
            just
            prek
            python
            ruff
            uv
          ];

          shellHook = ''
            export UV_PYTHON="${python}/bin/python3"
            export UV_PYTHON_DOWNLOADS=never
            export PATH="$PWD/.venv/bin:$PATH"

            echo "Run 'uv sync --dev' to install Python dependencies into .venv."
            echo "Then use 'just lint', 'just test', or 'just all'."
          '';
        };
      }
    );
}
