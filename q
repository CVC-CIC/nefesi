WC(1)                                                   User Commands                                                   WC(1)

NNAAMMEE
       wc - print newline, word, and byte counts for each file

SSYYNNOOPPSSIISS
       wwcc [_O_P_T_I_O_N]... [_F_I_L_E]...
       wwcc [_O_P_T_I_O_N]... _-_-_f_i_l_e_s_0_-_f_r_o_m_=_F

DDEESSCCRRIIPPTTIIOONN
       Print  newline, word, and byte counts for each FILE, and a total line if more than one FILE is specified.  A word is a
       non-zero-length sequence of characters delimited by white space.

       With no FILE, or when FILE is -, read standard input.

       The options below may be used to select which counts are printed, always in the following order: newline, word,  char‐
       acter, byte, maximum line length.

       --cc, ----bbyytteess
              print the byte counts

       --mm, ----cchhaarrss
              print the character counts

       --ll, ----lliinneess
              print the newline counts

       ----ffiilleess00--ffrroomm=_F
              read  input from the files specified by NUL-terminated names in file F; If F is - then read names from standard
              input

       --LL, ----mmaaxx--lliinnee--lleennggtthh
              print the maximum display width

       --ww, ----wwoorrddss
              print the word counts

       ----hheellpp display this help and exit

       ----vveerrssiioonn
              output version information and exit

AAUUTTHHOORR
       Written by Paul Rubin and David MacKenzie.

RREEPPOORRTTIINNGG BBUUGGSS
       GNU coreutils online help: <http://www.gnu.org/software/coreutils/>
       Report wc translation bugs to <http://translationproject.org/team/>

CCOOPPYYRRIIGGHHTT
       Copyright  ©   2016   Free   Software   Foundation,   Inc.    License   GPLv3+:   GNU   GPL   version   3   or   later
       <http://gnu.org/licenses/gpl.html>.
       This  is  free software: you are free to change and redistribute it.  There is NO WARRANTY, to the extent permitted by
       law.

SSEEEE AALLSSOO
       Full documentation at: <http://www.gnu.org/software/coreutils/wc>
       or available locally via: info '(coreutils) wc invocation'

GNU coreutils 8.25                                      February 2017                                                   WC(1)
