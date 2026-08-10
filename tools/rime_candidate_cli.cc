#include <rime_api.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

struct Options {
  std::string shared_data;
  std::string user_data;
  std::string prebuilt_data;
  std::string schema = "luna_pinyin";
  int max_candidates = 10;
};

bool ParseOptions(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    std::string argument = argv[i];
    if (i + 1 >= argc) {
      return false;
    }
    std::string value = argv[++i];
    if (argument == "--shared-data") {
      options->shared_data = value;
    } else if (argument == "--user-data") {
      options->user_data = value;
    } else if (argument == "--prebuilt-data") {
      options->prebuilt_data = value;
    } else if (argument == "--schema") {
      options->schema = value;
    } else if (argument == "--max-candidates") {
      options->max_candidates = std::stoi(value);
    } else {
      return false;
    }
  }
  return !options->shared_data.empty() && !options->user_data.empty() &&
         !options->prebuilt_data.empty() && options->max_candidates > 0;
}

std::string SafeField(const char* value) {
  std::string result = value ? value : "";
  std::replace(result.begin(), result.end(), '\t', ' ');
  std::replace(result.begin(), result.end(), '\n', ' ');
  std::replace(result.begin(), result.end(), '\r', ' ');
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    std::cerr << "usage: rime_candidate_cli --shared-data DIR --user-data DIR "
                 "--prebuilt-data DIR --schema ID --max-candidates K\n";
    return 2;
  }

  RimeTraits traits = {};
  traits.data_size = sizeof(RimeTraits) - sizeof(traits.data_size);
  traits.shared_data_dir = options.shared_data.c_str();
  traits.user_data_dir = options.user_data.c_str();
  traits.prebuilt_data_dir = options.prebuilt_data.c_str();
  traits.staging_dir = options.prebuilt_data.c_str();
  traits.distribution_name = "Phase 4B Rime Coverage Adapter";
  traits.distribution_code_name = "phase4b-rime";
  traits.distribution_version = "1";
  traits.app_name = "rime.phase4b_coverage";
  traits.min_log_level = 3;
  traits.log_dir = "";

  RimeApi* api = rime_get_api();
  api->setup(&traits);
  api->initialize(&traits);
  if (api->start_maintenance(false)) {
    api->join_maintenance_thread();
  }

  RimeSessionId session = api->create_session();
  if (!session || !api->select_schema(session, options.schema.c_str())) {
    std::cerr << "failed to create Rime session or select schema "
              << options.schema << "\n";
    if (session) api->destroy_session(session);
    api->finalize();
    return 3;
  }

  std::string input;
  while (std::getline(std::cin, input)) {
    api->clear_composition(session);
    if (!input.empty()) {
      api->simulate_key_sequence(session, input.c_str());
    }
    RimeCandidateListIterator iterator = {};
    int emitted = 0;
    if (api->candidate_list_begin(session, &iterator)) {
      while (emitted < options.max_candidates &&
             api->candidate_list_next(&iterator)) {
        if (emitted++) std::cout << '\t';
        std::cout << SafeField(iterator.candidate.text);
      }
      api->candidate_list_end(&iterator);
    }
    std::cout << '\n' << std::flush;
  }

  api->destroy_session(session);
  api->finalize();
  return 0;
}

