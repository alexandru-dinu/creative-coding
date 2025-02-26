CPPFLAGS=-Wall -Wextra -O2 -g -std=c++17
LIBS=

.PHONY: clean

%.o: %.cpp
	g++ $(CPPFLAGS) -c $^ -o $@
clean:
	find . -exec file {} \; | grep -i elf | cut -d":" -f1 | tee /dev/tty | xargs --no-run-if-empty rm
