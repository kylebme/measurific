cmake_minimum_required(VERSION 3.21)
project(inspection_camera_bridge VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ARAVIS REQUIRED IMPORTED_TARGET aravis-0.8>=0.8.23)

add_executable(inspection-camera-bridge src/main.c src/focus.c)
target_link_libraries(inspection-camera-bridge PRIVATE PkgConfig::ARAVIS)
# glibc keeps the math functions in a separate libm that must be linked explicitly.
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
  target_link_libraries(inspection-camera-bridge PRIVATE ${MATH_LIBRARY})
endif()
target_compile_options(inspection-camera-bridge PRIVATE -Wall -Wextra -Wpedantic)

if(APPLE)
  set_target_properties(inspection-camera-bridge PROPERTIES INSTALL_RPATH "@loader_path/../lib")
elseif(UNIX)
  set_target_properties(inspection-camera-bridge PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
endif()

install(TARGETS inspection-camera-bridge RUNTIME DESTINATION bin)

include(CTest)
if(BUILD_TESTING)
  add_executable(focus-test tests/focus.c src/focus.c)
  if(MATH_LIBRARY)
    target_link_libraries(focus-test PRIVATE ${MATH_LIBRARY})
  endif()
  target_compile_options(focus-test PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native-focus COMMAND focus-test)
endif()
