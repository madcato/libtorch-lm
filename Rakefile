NPROC = `nproc`.to_i

class OS
  def self.UNIX?
    RUBY_PLATFORM != 'x86_64-darwin19'
  end
  def self.macOS?
    RUBY_PLATFORM == 'x86_64-darwin19'
  end
end

def runUNIX
  mkdir_p 'build'
  Dir.chdir 'build'
  sh "cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch -DTorch_DIR=/opt/libtorch/share/cmake/Torch -Wno-dev .. && cmake --build . --config Release -- -j #{NPROC}"
  Dir.chdir '..'
end

def runMacOS
  mkdir_p 'build'
  Dir.chdir 'build'  
  sh "cmake -GXcode -DCMAKE_PREFIX_PATH=/opt/libtorch -DTorch_DIR=/opt/libtorch/share/cmake/Torch -Wno-dev .. && cmake --build . --config Release  -- -j #{NPROC}"
  Dir.chdir '..'
end

# directory 'build'

task :default => :build

desc 'Build aai and libs'
task :build do
  runUNIX if OS.UNIX?
  runMacOS if OS.macOS?
end

task :clean do
  rm_rf 'build'
end

task :noop do
end
