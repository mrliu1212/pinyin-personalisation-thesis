RIME_PREFIX ?= $(shell brew --prefix librime)
SDKROOT ?= $(shell xcrun --show-sdk-path)
CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -isysroot $(SDKROOT) -isystem $(SDKROOT)/usr/include/c++/v1
RIME_ADAPTER := .build/rime_candidate_cli

.PHONY: rime-adapter
rime-adapter: $(RIME_ADAPTER)

$(RIME_ADAPTER): tools/rime_candidate_cli.cc
	mkdir -p .build
	$(CXX) $(CXXFLAGS) -I$(RIME_PREFIX)/include $< \
		-L$(RIME_PREFIX)/lib -lrime -Wl,-rpath,$(RIME_PREFIX)/lib -o $@
