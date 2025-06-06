	subroutine upstr (str, len)
c
c	upstr converts the character string str to upper case.
c	len is the number of characters to convert, not to exceed the
c	actual length of str.
c
c	author: fred klein (u.s.g.s)
c
	character		str*(*)
	integer			i
	integer			j
	integer			len

	do 2 i = 1, len
	  j = ichar(str(i:i))
	  if (j .gt. 96 .and. j .lt. 123) str(i:i) =  char(j - 32)
2	continue
	return
	end

