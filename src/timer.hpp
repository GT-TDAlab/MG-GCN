#pragma once

#include <chrono>
#include <string>
#include <mutex>
#include <iostream>

class timer {
	static std::mutex iomutex;
	const std::string name;
	const std::chrono::high_resolution_clock::time_point start;
	std::mutex &mtx;
	std::ostream &os;
public:
	timer(std::string s, std::mutex &_mtx = iomutex, std::ostream &_os = std::cerr) : name(s), start(std::chrono::high_resolution_clock::now()), mtx(_mtx), os(_os) {
		#ifdef LOG
		std::lock_guard lock(mtx);
		os << name << " has started" << std::endl;
		#endif
	}
	~timer() {
		#ifdef LOG
		std::lock_guard lock(mtx);
		os << name << " took " << time() * 1000 << "ms" << std::endl;
		#endif
	}
	double time() const {
		return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start).count();
	}
};

std::mutex timer::iomutex;