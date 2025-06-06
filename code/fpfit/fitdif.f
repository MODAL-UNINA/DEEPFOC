c
	program fitdif
c
c Requires two fpfit summary files.  The first file ("free.fps") is assumed
c to be an unconstrained suite of solutions, and the second file ("fix.fps")
c to be computed by constraining the solution space using the "dir", "dip", 
c and "rak" commands of fpfit.  
c 
c The program then compares the two files and reports whether the misfit score
c for each focal mechanism in the "fix.fps" file is within the 90% confidence region
c of the "free.fps" misfit score.
c
c N.B.  You must have a version of fpfit that outputs the solution misfit value +
c       the 90% confidence estimate in columns 103-107.  Older versions of the
c       code report another parameter in this field
c
c The program produces three files that are not very imaginatively named.  
c "notsame.fps" are earthquakes for which the constrained solution is outside
c               the 90% limits of the unconstrained solutions 
c "same.fps" are earthquakes for which the constrained solution is within 
c               the 90% limits of the unconstrained solutions 
c "fitdif.report" indicates for each earthquake whether the two solutions are the
c               same or not, where "same" is defined by the 90% confidence region.
c
c The program also allows you some slack in the 90% confidence region.  It prompts
c for an enter extra increment for the 90% confidence region.  The default is 0.0.
c
c If you get the message 'missing event somewhere', it means the program didn't 
c find an event in the "free.fps" where it expected it.  It almost always results
c from a multiple solution line in that file.  I usually edit out the multiple solution
c and try again. This is not very satisfactory, but it works. 
c
c Send complaints to David Oppenheimer (oppen@alum.wr.usgs.gov)
c 1/10/92
c
	character*129	line1
	character*129	line2
	character*50	filnam
	character*4	same
	real		askr
	real		fitinc
	real		fit2
	real		fit90
	fitinc = 0.
	fitinc = askr ('enter extra increment for fit90-fit2', fitinc)
10	filnam = 'free.fps'
	call askc ('enter filename of "free" fps', filnam)
	open (1, file = filnam, status = 'old', err = 10)
20	filnam = 'fix.fps'
	call askc ('enter filename of "fix" fps', filnam)
        open (2, file = filnam, status = 'old', err = 20)
	open (3, file = 'same.fps', status = 'unknown')
	open (4, file = 'notsame.fps', status = 'unknown')
	open (7, file = 'fitdif.report', status = 'unknown')
	write (7, 21)
21	format (1x, 'event', t50, 'fit90', 6x, 'fit2', /)
	iflag = 0
	read (1, '(a)', end = 100) line1
	read (2, '(a)', end = 100) line2
25	if (line1(1:43) .eq. line2(1:43)) then
	  read (line2, '(t94,f4.2)') fit2
	  read (line1, '(t103,f5.2)') fit90
	  if ((fit90 + fitinc - fit2) .ge. 0.) then
	    write (3, '(a)') line1
	    same = 'same'
	  else
	    write (4, '(a)') line1
	    same = ' not'
	  end if
	  write (7, 30) line1(1:43), fit90, fit2, same
30	  format (1x, a, 2(5x, f5.2), 5x, a)
	  iflag = 1
	  read (1, '(a)', end = 100) line1
	else
	  if (iflag .eq. 0) then
	    print *, 'missing event somewhere'
	    print *, line1(1:43)
	    stop
	  else
	    iflag = 0
	  end if
	  read (2, '(a)', end = 100) line2
	end if
	goto 25
100	close (1)
	close (2)
	close (3)
	close (4)
	close (7)
	stop
	end
      real function askr (prompt, dflt)
c
c  askr prompts then reads a real value from the terminal.
c  the default value is returned on a cr response.
c
      real              dflt                            
c							! default supplied on carriage return and displayed in prompt
      character         prompt*(*)                      
c							! prompt string

      integer           i                               
c							! loop index
      integer           j                               
c							! loop index
      integer           leng                            
c							! function
      character         temp*20                         
c							! scratch
      integer		ounit
c							! logical unit for output (0 for UNIX, 6 for VMS)

      parameter (ounit = 0)
      write (temp, 10) dflt
10    format (g20.5)
      do 20 i = 1, 20
        if (temp(i:i) .ne. ' ') goto 30
20    continue
30    do 40 j = 20, 1, -1
        if (temp(j:j) .ne. ' ') goto 50
40    continue
50    write (ounit, 60) prompt, temp(i:j)
60    format (1x, a, ' [cr = ', a, ']? ', $)
      read (5, '(a)', err = 50, end = 70) temp
      if (leng(temp) .gt. 0) then
        read (temp, *, err = 50) askr
      else
        askr = dflt
      end if
70    return
      end

      subroutine askc (prompt, string)
c
c  askc prompts then reads a character string from the terminal.
c  the original value is unchanged by a cr response.
c
      character         prompt*(*)                      
c							! prompt string
      character         string*(*)                      
c							! character response, or original string on cr.

      character         temp*80                         
c							! scratch
      integer           leng                            
c							! function
      integer           nch                             
c							! number of characters
      integer		ounit
c							! logical unit for output (0 for UNIX, 6 for VMS)

      parameter (ounit = 0)
      nch = leng(string)
10    write (ounit, 20) prompt
20    format (1x, a)
      if (nch .lt. 20) then
        write (ounit, 30) string(1:nch)
30      format (' [cr = ', a, ']? ', $)
      else
        write (ounit, 40) string(1:nch)
40      format (' [cr = ', a, ']?')
      end if
      read (5, '(a)', err = 10, end = 50) temp
      if (leng(temp) .gt. 0) string = temp
50    return
      end

      integer function leng (string)
c
c the non-blank length of string whose physical length is maxlen
c (returns the position of the last non-blank character)
c
      character         string*(*)                      
c							! string
c
      integer           i                               
c							! character position
      integer           maxlen                          
c							! length of string

      maxlen = len(string)
      do 10 i = maxlen,1,-1
        if (string(i:i) .ne. ' ') goto 20
10    continue
      i = 0
20    leng = i
      return
      end
