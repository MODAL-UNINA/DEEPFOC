      subroutine intero (iunit, ansn, answ, ansd, ansc, ncfrm, ncup,
     1 ncdwn, option, nskip, funit, date, filnam, y2k)
c
c get user options
c
      implicit none
c
      character*1       ansc                            
c                                                       ! flag: y(n)=do (not) plot color beach balls
      character*1       ansd                            
c                                                       ! flag: y(n)=do (not) plot discrepancy report
      character*1       ansn                            
c                                                       ! flag: y(n)=do (not) plot station names adjacent to first motions
      character*1       answ                            
c                                                       ! flag: y(n)=do (not) plot symbol size and names proportional to weight
      character*11      date                            
c                                                       ! requested event date and origin time
      character*50      filfps                          
c                                                       ! file name of .fps file
      character*50      filnam                          
c                                                       ! file name of data
      integer           funit                           
c                                                       ! logical unit ".fps" file
      integer           ios                             
c                                                       ! i/o status descriptor
      integer           iunit                           
c                                                       ! logical unit ".pol" file
      integer		jask
c							! function
      character*100     line                          
c							! test line to determine which .pol format (y1k or y2k)
      integer           ncfrm
c                                                       ! frame pen color
      integer           ncup
c                                                       ! up pen color
      integer           ncdwn
c                                                       ! down pen color
      integer           nskip                           
c                                                       ! number of solutions to skip
      character*1       option                          
c                                                       ! plot option
      integer           ounit
c                                                       ! logical unit for output 
      logical           y2k
c							! T(F) = .pol file format is (not) y2k compatible
c
      parameter (ounit = 6)

10    filnam = 'none'
      call askc ('Enter name of first ".pol" file:  ', filnam)
      open (iunit, file = filnam, status = 'old', blank
     1 = 'zero', iostat = ios)
      if (ios .ne. 0 .or. filnam .eq. 'none') then
        write (ounit, '(a)') 'Error opening file - try again'
        goto 10
      else
c
c  is this y1k or y2k format?
c
        read (iunit, '(a)') line
        read (iunit, '(a)') line
	if (line(15:15) .eq. '.' .and. line(34:34) .eq. '.') then
	  y2k = .false.
	elseif (line(17:17) .eq. '.' .and. line(36:36) .eq. '.') then
	  y2k = .true.
	else
	  write (ounit, '(a)') 
     1 'Unable to determine whether file is y1k or y2k format'
	  stop
        end if
      end if
      rewind (iunit)
40    ansn = 'n'
      call askc ('Plot station names (y or n)?  ', ansn)
      if (ansn .ne. 'y' .and. ansn .ne. 'n') then
        write (ounit, '(a)')
     & '**** Please answer "y" or "n"; try again ****'
        goto 40
      end if
45    answ = 'y'
      call askc ('Plot symbol size proportional to weights (y or n)?  ',
     & answ)
      if (answ .ne. 'y' .and. answ .ne. 'n') then
        write (ounit, '(a)')
     & '**** Please answer "y" or "n"; try again ****'
        goto 45
      end if
46    ansd = 'y'
      call askc ('Plot discrepancy report (y or n)?  ', ansd)
      if (ansd .ne. 'y' .and. ansd .ne. 'n') then
        write (ounit, '(a)')
     & '**** Please answer "y" or "n"; try again ****'
        goto 46
      end if
47    ansc = 'n'
c      call askc ('Plot color beachballs (y or n)?  ', ansc)
      if (ansc .ne. 'y' .and. ansc .ne. 'n') then
        write (ounit, '(a)')
     & '**** Please answer "y" or "n"; try again ****'
        goto 47
      else if (ansc .eq. 'y') then
	ncfrm = 7
	ncfrm = jask ('Enter title and net color', ncfrm)
	ncup = 9
	ncup = jask ('Enter compressional color', ncup)
	ncdwn = 6
	ncdwn = jask ('Enter dilatational color', ncdwn)
      end if
      write (ounit, '(a)') ' '
      write (ounit, '(1x, a)')  'Menu of plot options'
      write (ounit, '(1x, a)')  'a = Plot all mechanisms'
      write (ounit, '(1x, a)')
     $ 'd = Request mechanisms by date & origin time'
      write (ounit, '(1x, a)')  'f = Request mechanisms from .fps file'
      write (ounit, '(1x, a)')  
     & 'n = Request mechanisms by sequence number (including multiples)'
      write (ounit, '(a)') ' '
50    option = 'a'
      call askc ('Enter plot sequence option:  ', option)
      if (option .ne. 'a' .and. option .ne. 'd' .and. option .ne. 'f'
     & .and. option .ne. 'n') then
        write (ounit, 60) '**** Unknown option; please try again ****'
60      format (//, a, /)
        goto 50
      else if (option .eq. 'a') then
70      nskip = 0
        nskip = jask ('Enter number of mechanisms to skip (including mul
     &tiple solutions):  ', nskip)
        if (nskip .lt. 0) then
          write (ounit, 60) '**** Invalid number; try again ****'
          goto 70
        end if
      else if (option .eq. 'd') then
        date = '860531 1531'
      else if (option .eq. 'f') then
71      filfps = 'none'
        call askc ('Enter name of ".fps" file:  ', filfps)
        if (filfps .ne. 'none') then
          open (funit, file = filfps, status = 'old', blank
     & = 'zero', iostat = ios)
        end if
        if (ios .ne. 0 .or. filfps .eq. 'none') then
          write (ounit, '(a)') 'Error opening file - try again'
        goto 71
        end if
      end if
      return
      end
