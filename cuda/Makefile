.PHONY: build run configure clean distclean

build: build/build.ninja
	cmake --build build

run: build
	./build/gas-sim $(ARGS)

configure build/build.ninja:
	cmake -B build

clean:
	if [ -f build/build.ninja ]; then cmake --build build --target clean; fi

distclean:
	rm -rf build/
