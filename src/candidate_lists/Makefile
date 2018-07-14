# Makefile
# Project maintenance for saucy
#
# by Paul T. Darga <pdarga@umich.edu>
# and Mark Liffiton <liffiton@umich.edu>
# and Hadi Katebi <hadik@umich.edu>
#
# Copyright (C) 2004, The Regents of the University of Michigan
# See the LICENSE file for details.

CC=g++
#CFLAGS=-ansi -pedantic -Wall -O0 -ggdb
CFLAGS=-ansi -pedantic -Wall -O3
LDLIBS=-lz -lpthread
CXX=g++
CXXFLAGS=-std=c++17 -Ofast


.PHONY : all clean

all : saucy shatter

saucy : main.o saucy.o saucyio.o util.o
shatter : shatter.o saucy.o saucyio.o util.o
colour_signature : colour_signature.o saucy.o util.o

main.o shatter.o saucy.o saucyio.o colour_signature.o: saucy.h
main.o shatter.o saucyio.o colour_signature.o : amorph.h
main.o shatter.o util.o colour_signature.o : util.h
main.o shatter.o saucyio.o colour_signature.o : platform.h

clean :
	rm -f saucy shatter *.o
