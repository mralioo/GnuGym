#
# Copyright (C) 2013, 2016 Bastian Bloessl <bloessl@ccs-labs.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

########################################################################
# Setup library
########################################################################
include(GrPlatform) #define LIB_SUFFIX

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/generate_helper.py "
#!${PYTHON_EXECUTABLE}

import sys, os, re
sys.path.append('${GR_RUNTIME_PYTHONPATH}')
os.environ['srcdir'] = '${CMAKE_CURRENT_SOURCE_DIR}'
os.chdir('${CMAKE_CURRENT_BINARY_DIR}')

if __name__ == '__main__':
    sys.path.append('${CMAKE_CURRENT_SOURCE_DIR}/../python')
    import build_utils
    root, inp = sys.argv[1:3]
    for sig in sys.argv[3:]:
        name = re.sub ('X+', sig, root)
        d = build_utils.standard_dict(name, sig, 'gnugym')
        build_utils.expand_template(d, inp, '_impl')
")


macro(expand_cc_h_impl root)
  #make a list of all the generated files
  unset(expanded_files_cc_impl)
  unset(expanded_files_h_impl)
  foreach(sig ${ARGN})
    string(REGEX REPLACE "X+" ${sig} name ${root})
    list(APPEND expanded_files_cc_impl ${CMAKE_CURRENT_BINARY_DIR}/${name}_impl.cc)
    list(APPEND expanded_files_h_impl ${CMAKE_CURRENT_BINARY_DIR}/${name}_impl.h)
    list(APPEND expanded_files_h ${CMAKE_CURRENT_BINARY_DIR}/../include/${name}.h)
  endforeach(sig)

  #create a command to generate the _impl.cc files
  add_custom_command(
    OUTPUT ${expanded_files_cc_impl}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${root}_impl.cc.t
    COMMAND ${PYTHON_EXECUTABLE} ${PYTHON_DASH_B}
    ${CMAKE_CURRENT_BINARY_DIR}/generate_helper.py
    ${root} ${root}_impl.cc.t ${ARGN}
  )

  #create a command to generate the _impl.h files
  add_custom_command(
    OUTPUT ${expanded_files_h_impl}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${root}_impl.h.t
    COMMAND ${PYTHON_EXECUTABLE} ${PYTHON_DASH_B}
    ${CMAKE_CURRENT_BINARY_DIR}/generate_helper.py
    ${root} ${root}_impl.h.t ${ARGN}
  )

  #make _impl.cc source files depend on headers to force generation
  set_source_files_properties(${expanded_files_cc_impl}
    PROPERTIES OBJECT_DEPENDS "${expanded_files_h_impl}"
  )

  #make _impl.h source files depend on headers to force generation
  set_source_files_properties(${expanded_files_h_impl}
    PROPERTIES OBJECT_DEPENDS "${expanded_files_h}"
  )

  #install rules for the generated cc files
  list(APPEND generated_sources ${expanded_files_cc_impl})
endmacro(expand_cc_h_impl)

########################################################################
# Invoke macro to generate various sources
########################################################################
add_custom_target(gnugym_generated_sources DEPENDS ${generated_sources})
add_dependencies(gnugym_generated_sources gnugym_generated_includes)

########################################################################
# Setup library
########################################################################
list(APPEND gr_gnugym_sources
    ${generated_sources}
    constellations_impl.cc
    gnugym_rssi_cb_impl.cc
    gnugym_snr_cb_impl.cc
    equalizer/base.cc
    equalizer/comb.cc
    equalizer/ls.cc
    equalizer/lms.cc
    equalizer/sta.cc
    gnugym_snrdebug_ff_impl.cc
    gnugym_moving_average_vect_ff_impl.cc
    gnugym_parse_seqnr_impl.cc
)

set(GR_REQUIRED_COMPONENTS RUNTIME FILTER DIGITAL)

list(APPEND gr_gnugym_libs
    ${Boost_LIBRARIES}
    ${GNURADIO_ALL_LIBRARIES}
    ${LOG4CPP_LIBRARIES}
)

include_directories(${GNURADIO_ALL_INCLUDE_DIRS})

add_library(gnuradio-gnugym SHARED ${gr_gnugym_sources})
add_dependencies(gnuradio-gnugym gnugym_generated_sources)
target_link_libraries(gnuradio-gnugym ${gr_gnugym_libs})
set_target_properties(gnuradio-gnugym PROPERTIES DEFINE_SYMBOL "gnuradio_gnugym_EXPORTS")

########################################################################
# Install built library files
########################################################################
install(TARGETS gnuradio-gnugym
    LIBRARY DESTINATION lib${LIB_SUFFIX} # .so/.dylib file
    ARCHIVE DESTINATION lib${LIB_SUFFIX} # .lib file
    RUNTIME DESTINATION bin              # .dll file
)
