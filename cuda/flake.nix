{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = inputs: let
    system = "x86_64-linux";
    pkgs = import inputs.nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };
    cuda = pkgs.cudaPackages_12_9;
    clangTools = (pkgs.clang-tools.override {
      clang = pkgs.llvmPackages.clang.override {
        gccForLibs = cuda.backendStdenv.cc.cc;
      };
    }).overrideAttrs (old: {
      postInstall = (old.postInstall or "") + ''
        ln -s ${pkgs.llvmPackages.clang-unwrapped}/bin/nvptx-arch "$out/bin/nvptx-arch"
      '';
    });
    toolkit = pkgs.symlinkJoin {
      name = "cuda-dev-toolkit";
      paths = [cuda.cuda_nvcc cuda.cuda_cudart cuda.cccl];
    };
  in {
    devShells.${system}.default = (pkgs.mkShell.override {stdenv = cuda.backendStdenv;}) {
      nativeBuildInputs = [
        pkgs.cmake
        pkgs.ninja
        pkgs.pkg-config
        clangTools
        toolkit
        cuda.cuda_gdb
        cuda.cuda_sanitizer_api
        cuda.nsight_systems
        cuda.nsight_compute
      ];

      buildInputs = [cuda.cuda_cudart cuda.cccl cuda.libcurand.include];

      CUDA_PATH = "${toolkit}";
      CUDAToolkit_ROOT = "${toolkit}";
      CUDACXX = "${toolkit}/bin/nvcc";
      CUDAHOSTCXX = "${cuda.backendStdenv.cc}/bin/g++";
      CUDAARCHS = "native";
      CMAKE_GENERATOR = "Ninja";

      shellHook = ''
        mkdir -p .direnv
        ln -sfT "${toolkit}" .direnv/cuda
        export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath [cuda.cuda_cudart]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      '';
    };
  };
}
