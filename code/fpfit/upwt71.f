	subroutine upwt71 (card1, card2, isflag, icflag, iaflag)

c compare two multiple cards and increases reading weight of secondary if required

	character*(*)	card1				
c							! (input/output) primary phase card
	character*(*)	card2				
c							! (input/output) secondary card
	integer		iaflag				
c							! (input) 1(2)=do(not) retain amplitude from secondary if none on primary
	integer		icflag				
c							! (input)  1(2)=do(not) retain coda from secondary if none on primary
	integer		isflag				
c							! (input)  1(2)=s on vrtcl in scndry will (not) be rtaind if not in prmry
c
	integer		aer1				
c							! flag: 0(1)=awt1 is(not) an integer
	integer		aer2				
c							! flag: 0(1)=awt2 is(not) an integer
	real		amp1				
c							! peak-to-peak amplitude on primary card
	real		amp2				
c							! peak-to-peak amplitude on secondary card
	integer		awt1				
c							! amplitude wt of primary card
	integer		awt2				
c							! amplitude wt of secondary card
	integer		cer1				
c							! flag: 0(1)=cwt1 is(not) an integer
	integer		cer2				
c							! flag: 0(1)=cwt2 is(not) an integer
	real		coda1				
c							! peak-to-peak coda on primary card
	real		coda2				
c							! peak-to-peak coda on secondary card
	integer		cwt1				
c							! coda wt of primary card
	integer		cwt2				
c							! coda wt of secondary card
	integer		per1				
c							! flag: 0(1)=pwt1 is(not) an integer
	integer		per2				
c							! flag: 0(1)=pwt2 is(not) an integer
	integer		pwt1				
c							! p-wt of primary card
	integer		pwt2				
c							! p-wt of secondary card
	integer		ser1				
c							! flag: 0(1)=swt1 is(not) an integer
	integer		ser2				
c							! flag: 0(1)=swt2 is(not) an integer
	integer		swt1				
c							! s-wt of primary card
	integer		swt2				
c							! s-wt of secondary card
c
	call rdwt (card1, 8, pwt1, per1)
	call rdwt (card1, 40, swt1, ser1)
	call rdwt (card1, 77, awt1, aer1)
	call rdwt (card1, 76, cwt1, cer1)
	call rdwt (card2, 8, pwt2, per2)
	call rdwt (card2, 40, swt2, ser2)
	call rdwt (card2, 77, awt2, aer2)
	call rdwt (card2, 76, cwt2, cer2)
	read (card1(45:47), '(f3.0)') amp1
	read (card2(45:47), '(f3.0)') amp2
	read (card1(72:75), '(f4.0)') coda1
	read (card2(72:75), '(f4.0)') coda2
c-------------------------------------------------------------------------------
c check p weight
c-------------------------------------------------------------------------------
	if (pwt2 .lt. 4 .and. pwt1 .lt. 4
     & .and. card1(5:6) .ne. '  ' .and. card2(5:6) .ne. '  ') then
          pwt2 = pwt2 + 5
          write (card2(8:8), '(i1)') pwt2
        end if
c-------------------------------------------------------------------------------
c check s weight
c-------------------------------------------------------------------------------
	if ((swt2 .lt. 4 .and. card2(37:38) .ne. '  ') .and.
     & ((isflag .eq. 1 .and. card1(37:38) .ne. '  ' .and. swt1 .lt. 4)
     & .or. (isflag .eq. 2 .and. card1(37:38) .eq. '  '))) then
          swt2 = swt2 + 5
          write (card2(40:40), '(i1)') swt2
        end if
        if (card1(9:9) .eq. card2(9:9)) then
c-------------------------------------------------------------------------------
c check amplitude weight
c-------------------------------------------------------------------------------
	  if ((amp2 .ne. 0. .and. awt2 .lt. 4) .and.
     & ((iaflag .eq. 1 .and. amp1 .ne. 0. .and. awt1 .lt. 4)
     & .or. (iaflag .eq. 2 .and. amp1 .eq. 0.)))
     & write (card2(77:77), '(i1)') awt2 + 5
c-------------------------------------------------------------------------------
c check coda duration weight
c-------------------------------------------------------------------------------
	  if ((coda2 .ne. 0. .and. cwt2 .lt. 4) .and.
     & ((icflag .eq. 1 .and. coda1 .ne. 0. .and. cwt1 .lt. 4)
     & .or. (icflag .eq. 2 .and. coda1 .eq. 0.)))
     & write (card2(76:76), '(i1)') cwt2 + 5
        end if
c-------------------------------------------------------------------------------
        
c-------------------------------------------------------------------------------
c add weight-out code
c-------------------------------------------------------------------------------
c       if (pwt2 .ge. 4 .and. swt2 .ge. 4) write(card2(25:25),'(a)') '*'
	return
	end
