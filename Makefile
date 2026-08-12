.PHONY: rime-adapter

ifeq ($(OS),Windows_NT)
RIME_ADAPTER := .build/rime_candidate_cli.exe

rime-adapter: tools/rime_candidate_cli.cc tools/build_rime_adapter.ps1
	powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/build_rime_adapter.ps1 -RimePrefix "$(RIME_PREFIX)"
else
RIME_PREFIX ?= $(shell brew --prefix librime)
SDKROOT ?= $(shell xcrun --show-sdk-path)
CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -isysroot $(SDKROOT) -isystem $(SDKROOT)/usr/include/c++/v1
RIME_ADAPTER := .build/rime_candidate_cli

rime-adapter: $(RIME_ADAPTER)

$(RIME_ADAPTER): tools/rime_candidate_cli.cc
	mkdir -p .build
	$(CXX) $(CXXFLAGS) -I$(RIME_PREFIX)/include $< \
		-L$(RIME_PREFIX)/lib -lrime -Wl,-rpath,$(RIME_PREFIX)/lib -o $@
endif
