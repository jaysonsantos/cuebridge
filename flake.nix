{
  description = "CueBridge package, overlay, and development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      mkPythonSet =
        pkgs:
        let
          python = pkgs.python312;
          baseSet = pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          };
          projectOverlay = workspace.mkPyprojectOverlay {
            sourcePreference = "wheel";
          };
          pyprojectOverrides =
            final: prev:
            let
              inherit (final) resolveBuildSystem;
            in
            {
              cuebridge = prev.cuebridge.overrideAttrs (old: {
                nativeBuildInputs =
                  (old.nativeBuildInputs or [ ])
                  ++ resolveBuildSystem {
                    editables = [ ];
                  };

                passthru = (old.passthru or { }) // {
                  tests =
                    ((old.passthru or { }).tests or { })
                    // {
                      pytest =
                        let
                          virtualenv = final.mkVirtualEnv "cuebridge-test-env" {
                            cuebridge = [ "dev" ];
                          };
                        in
                        pkgs.stdenvNoCC.mkDerivation {
                          name = "${final.cuebridge.name}-pytest";
                          inherit (final.cuebridge) src;
                          nativeBuildInputs = [ virtualenv ];
                          dontConfigure = true;
                          buildPhase = ''
                            runHook preBuild
                            pytest
                            runHook postBuild
                          '';
                          installPhase = ''
                            runHook preInstall
                            mkdir -p $out
                            touch $out/passed
                            runHook postInstall
                          '';
                        };
                    };
                };
              });
            };
        in
        baseSet.overrideScope (
          lib.composeManyExtensions [
            pyproject-build-systems.overlays.default
            projectOverlay
            pyprojectOverrides
          ]
        );

      mkPackage =
        pkgs:
        let
          pythonSet = mkPythonSet pkgs;
          virtualenv = pythonSet.mkVirtualEnv "cuebridge-env" workspace.deps.default;
        in
        pkgs.runCommand "cuebridge" { } ''
          mkdir -p "$out/bin"
          ln -s "${virtualenv}/bin/cuebridge" "$out/bin/cuebridge"
        '';

    in
    {
      overlays.default =
        final: prev:
        let
          pythonSet = mkPythonSet prev;
          virtualenv = pythonSet.mkVirtualEnv "cuebridge-env" workspace.deps.default;
        in
        {
          cuebridge = prev.runCommand "cuebridge" { } ''
            mkdir -p "$out/bin"
            ln -s "${virtualenv}/bin/cuebridge" "$out/bin/cuebridge"
          '';
          cuebridge-unwrapped = pythonSet.cuebridge;
          cuebridge-python-set = pythonSet;
        };

      packages = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          pythonSet = mkPythonSet pkgs;
          package = mkPackage pkgs;
        in
        {
          default = package;
          cuebridge = package;
          cuebridge-unwrapped = pythonSet.cuebridge;
        }
      );

      apps = forAllSystems (
        system:
        let
          package = self.packages.${system}.default;
        in
        {
          default = {
            type = "app";
            program = "${package}/bin/cuebridge";
          };
          cuebridge = {
            type = "app";
            program = "${package}/bin/cuebridge";
          };
        }
      );

      checks = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          pythonSet = mkPythonSet pkgs;
          package = mkPackage pkgs;
        in
        {
          default = package;
          package = package;
          pytest = pythonSet.cuebridge.passthru.tests.pytest;
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          pythonSet = mkPythonSet pkgs;
          cxxRuntimePath = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ];
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              git
              gitleaks
              just
              pythonSet.python
              uv
            ];

            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
            };

            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT="$(git rev-parse --show-toplevel)"
              export PATH="$REPO_ROOT/.venv/bin:$PATH"
              export LD_LIBRARY_PATH="${cxxRuntimePath}:$LD_LIBRARY_PATH"

              echo "CueBridge Nix environment loaded."
              echo "Run 'uv sync --dev' to install Python dependencies into .venv."
              echo "Then use 'just lint', 'just test', 'just all', or 'cuebridge'."
            '';
          };
        }
      );
    };
}
