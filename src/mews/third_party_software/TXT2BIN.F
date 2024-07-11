      PROGRAM FMTWTH
C      
C     THIS PROGRAM READS A FORMATTED WEATHER FILE (WEATHER.FMT)
C     AND A FORMATTED INPUT FILE (FMTWTH.INP) AND WRITES A
C     PACKED BINARY DOE2 WEATHER FILE (WEATHER.BIN)
C
      DIMENSION CLN(12),GT(12),MDAYS(12),IDAT(1536),IWDID(5)
C     
      DATA MDAYS / 31,28,31,30,31,30,31,31,30,31,30,31 /
C      
      OPEN(UNIT=12,FILE='INPUT.DAT')
      OPEN (UNIT=11,FILE='WEATHER.FMT',blocksize=710000)
      OPEN (UNIT=10,FILE='WEATHER.BIN',FORM='UNFORMATTED',
     $      recordtype='variable',recl=6200,blocksize=148992)
C              
C     IWSZ           WORD SIZE          1 = 60-BIT, 2 = 30-BIT
C     IFTYP          FILE TYPE          1 = OLD, 2 = NORMAL (NO SOLAR),
C                                       3 = THE DATA HAS SOLAR
C     IWDID          LOCATION I.D.
C     IWYR           YEAR
C     WLAT           LATITUDE
C     WLONG          LONGITUDE
C     IWTZN          TIME ZONE NUMBER
C     IWSOL          SOLAR FLAG         FUNCTION OF IWSZ + IFTYP
C     CLN            CLEARNESS NO.
C     GT             GROUND TEMP.       (DEG R)
C     KMON           MONTH              (1-12)
C     KDAY           DAY OF MONTH
C     KH             HOUR OF DAY
C     WBT            WET BULB TEMP      (DEG F)
C     DBT            DRY BULB TEMP      (DEG F)
C     PATM           PRESSURE           (INCHES OF HG)
C     CLDAMT         CLOUD AMOUNT       (0 - 10)
C     ISNOW          SNOW FLAG          (1 = SNOWFALL)
C     IRAIN          RAIN FLAG          (1 = RAINFALL)
C     IWNDDR         WIND DIRECTION     (0 - 15; 0=N, 1=NNE, ETC)
C     HUMRAT         HUMIDITY RATIO     (LB H2O/LB AIR)
C     DENSTY         DENSITY OF AIR     (LB/CU FT)
C     ENTHAL         SPECIFIC ENTHALPY  (BTU/LB)
C     SOLRAD         TOTAL HOR. SOLAR   (BTU/HR-SQFT)
C     DIRSOL         DIR. NORMAL SOLAR  (BTU/HR-SQFT)
C     ICLDTY         CLOUD TYPE         (0 - 2)
C     WNDSPD         WIND SPEED         KNOTS
C
      IWSZ  = 0
      IFTYP = 0
      REWIND 12
      READ (12,9001,END=12) IWSZ,IFTYP
 9001 FORMAT(12X,I1,17X,I1)     
   12 REWIND 11
      READ (11,9002) (IWDID(I),I=1,5),IWYR,WLAT,WLONG,IWTZN,IWSOL
 9002 FORMAT(5A4,I5,2F8.2,2I5)     
      PRINT 3,(IWDID(I),I=1,5)
    3 FORMAT(' FMTWTH2: Processing weather data for: ',5A4)
      IF(IWSZ .eq. 0) THEN
        PRINT 1, IWSOL
    1   FORMAT('          Using file type found on WEATHER.FMT of ',I1)
      ELSE
        IWSOL = IWSZ + (IFTYP-1)*2 - 1
        PRINT 2, IWSOL
    2   FORMAT('          Using file type found on INPUT.TMP of ',I1)
      ENDIF
      PRINT 4
    4 FORMAT('          (5 = Solar data on file, 3= No solar on file)')
      READ (11,9003) (CLN(I),I=1,12)
      READ (11,9004) (GT(I),I=1,12)
 9003 FORMAT(12F6.2)      
 9004 FORMAT(12F6.1)
      DO 1000 IM=1,12
      IDE = MDAYS(IM)
      DO 1000 ID=1,IDE
      IRECXO = IM*2 + (ID-1)/16 - 1
      IDXO = MOD(ID-1,16) + 1
      DO 500 IH=1,24
      READ (11,9005) KMON, KDAY, KH, WBT, DBT, PATM, CLDAMT, ISNOW, 
     1               IRAIN, IWNDDR, HUMRAT, DENSTY, ENTHAL, SOLRAD,
     2               DIRSOL, ICLDTY, WNDSPD
 9005 FORMAT(3I2,2F5.0,F6.1,F5.0,2I3,I4,F7.4,F6.3,F6.1,2F7.1,I3,F5.0)      
      ISOL = INT(SOLRAD + .5)
      IDN = INT(DIRSOL + .5)
      IWET = INT(WBT+99.5)
      IDRY = INT(DBT+99.5)
      IPRES = INT(PATM*10.-149.5)
      ICLDAM = INT(CLDAMT)
      IWNDSP = INT(WNDSPD+0.5)
      IHUMRT = INT(HUMRAT*10000.+0.5)
      IDENS = INT(DENSTY*1000.-19.5)
      IENTH = INT(ENTHAL*2.0+60.5)
      IP1 = (IDXO-1)*96 + IH*4 - 3
      IDAT(IP1) = IPRES*65536 + IWET*256 + IDRY
      IDAT(IP1+1) = ISOL*1048576 + IDN*1024 + 
     1              ICLDAM*64 + ISNOW*32 + IRAIN*16 + IWNDDR 
      IDAT(IP1+2) = IHUMRT*128 + IDENS
      IDAT(IP1+3) = IENTH*2048 + ICLDTY*128 + IWNDSP
  500 CONTINUE      
      IF ((ID .NE. 16) .AND. (ID .NE. IDE))  GO TO 1000
      WRITE (10) IWDID,IWYR,WLAT,WLONG,IWTZN,IRECXO,IDE,CLN(IM),
     1           GT(IM),IWSOL,IDAT 
 1000 CONTINUE     
      END
      
