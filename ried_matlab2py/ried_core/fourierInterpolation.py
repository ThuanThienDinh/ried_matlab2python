import numpy as np


def fInterp_2D(img, newsz):
    sz = np.array(img.shape)
    newsz = np.array(newsz, dtype=int)

    if np.any(newsz == 0):
        return np.array([])

    incr = np.ones(2, dtype=int)

    for i in range(2):
        if newsz[i] < sz[i]:
            incr[i] = sz[i] // newsz[i] + 1
            newsz[i] *= incr[i]

    img_ip = np.zeros(tuple(newsz), dtype=complex)

    nyqst = np.ceil((sz + 1) / 2).astype(int)

    img_fft = (
        newsz[0] / sz[0]
        * newsz[1] / sz[1]
        * np.fft.fft2(img)
    )

    img_ip[
        :nyqst[0],
        :nyqst[1]
    ] = img_fft[
        :nyqst[0],
        :nyqst[1]
    ]

    img_ip[
        -(sz[0] - nyqst[0]):,
        :nyqst[1]
    ] = img_fft[
        nyqst[0]:,
        :nyqst[1]
    ]

    img_ip[
        :nyqst[0],
        -(sz[1] - nyqst[1]):
    ] = img_fft[
        :nyqst[0],
        nyqst[1]:
    ]

    img_ip[
        -(sz[0] - nyqst[0]):,
        -(sz[1] - nyqst[1]):
    ] = img_fft[
        nyqst[0]:,
        nyqst[1]:
    ]

    rm = sz % 2

    if rm[0] == 0 and newsz[0] != sz[0]:
        img_ip[nyqst[0] - 1, :] /= 2
        img_ip[
            nyqst[0] - 1 + newsz[0] - sz[0],
            :
        ] = img_ip[nyqst[0] - 1, :]

    if rm[1] == 0 and newsz[1] != sz[1]:
        img_ip[:, nyqst[1] - 1] /= 2
        img_ip[
            :,
            nyqst[1] - 1 + newsz[1] - sz[1]
        ] = img_ip[:, nyqst[1] - 1]

    img_ip = np.real(
        np.fft.ifft2(img_ip)
    )

    return img_ip[
        ::incr[0],
        ::incr[1]
    ]


def _fInterp_3D(img, newsz):
    sz = np.array(img.shape)
    newsz = np.array(newsz, dtype=int)

    if np.any(newsz == 0):
        return np.array([])

    incr = np.ones(3, dtype=int)

    for i in range(3):
        if newsz[i] < sz[i]:
            incr[i] = sz[i] // newsz[i] + 1
            newsz[i] *= incr[i]

    img_ip = np.zeros(tuple(newsz), dtype=complex)

    nyqst = np.ceil((sz + 1) / 2).astype(int)

    img_fft = (
        newsz[0] / sz[0]
        * newsz[1] / sz[1]
        * newsz[2] / sz[2]
        * np.fft.fftn(img)
    )

    x1 = slice(0, nyqst[0])
    x2 = slice(nyqst[0], sz[0])
    y1 = slice(0, nyqst[1])
    y2 = slice(nyqst[1], sz[1])
    z1 = slice(0, nyqst[2])
    z2 = slice(nyqst[2], sz[2])

    Xh = slice(-(sz[0] - nyqst[0]), None)
    Yh = slice(-(sz[1] - nyqst[1]), None)
    Zh = slice(-(sz[2] - nyqst[2]), None)

    img_ip[x1, y1, z1] = img_fft[x1, y1, z1]
    img_ip[Xh, y1, z1] = img_fft[x2, y1, z1]
    img_ip[x1, Yh, z1] = img_fft[x1, y2, z1]
    img_ip[x1, y1, Zh] = img_fft[x1, y1, z2]
    img_ip[Xh, Yh, z1] = img_fft[x2, y2, z1]
    img_ip[Xh, y1, Zh] = img_fft[x2, y1, z2]
    img_ip[x1, Yh, Zh] = img_fft[x1, y2, z2]
    img_ip[Xh, Yh, Zh] = img_fft[x2, y2, z2]

    rm = sz % 2

    if rm[0] == 0 and newsz[0] != sz[0]:
        img_ip[nyqst[0] - 1, :, :] /= 2
        img_ip[
            nyqst[0] - 1 + newsz[0] - sz[0],
            :,
            :
        ] = img_ip[nyqst[0] - 1, :, :]

    if rm[1] == 0 and newsz[1] != sz[1]:
        img_ip[:, nyqst[1] - 1, :] /= 2
        img_ip[
            :,
            nyqst[1] - 1 + newsz[1] - sz[1],
            :
        ] = img_ip[:, nyqst[1] - 1, :]

    if rm[2] == 0 and newsz[2] != sz[2]:
        img_ip[:, :, nyqst[2] - 1] /= 2
        img_ip[
            :,
            :,
            nyqst[2] - 1 + newsz[2] - sz[2]
        ] = img_ip[:, :, nyqst[2] - 1]

    img_ip = np.real(
        np.fft.ifftn(img_ip)
    )

    return img_ip[
        ::incr[0],
        ::incr[1],
        ::incr[2]
    ]


def _get_valid_part(img, input_sz, itp_fac, noip, idx, mirror_mode):
    doip = ~noip

    if img.ndim == 2:
        if noip[0] and noip[1]:
            return img

        if noip[0] and doip[1]:
            return img[
                :,
                idx[1]:idx[1] + itp_fac[1] * input_sz[1]
            ]

        if doip[0] and noip[1]:
            return img[
                idx[0]:idx[0] + itp_fac[0] * input_sz[0],
                :
            ]

        return img[
            idx[0]:idx[0] + itp_fac[0] * input_sz[0],
            idx[1]:idx[1] + itp_fac[1] * input_sz[1]
        ]

    if mirror_mode == "lateral":
        if noip[0] and noip[1]:
            return img

        if noip[0] and doip[1]:
            return img[
                :,
                idx[1]:idx[1] + itp_fac[1] * input_sz[1],
                :
            ]

        if doip[0] and noip[1]:
            return img[
                idx[0]:idx[0] + itp_fac[0] * input_sz[0],
                :,
                :
            ]

        return img[
            idx[0]:idx[0] + itp_fac[0] * input_sz[0],
            idx[1]:idx[1] + itp_fac[1] * input_sz[1],
            :
        ]

    if mirror_mode == "axial":
        if doip[2]:
            return img[
                :,
                :,
                idx[2]:idx[2] + itp_fac[2] * input_sz[2]
            ]
        return img

    if mirror_mode == "both":
        if noip[2]:
            if noip[0] and noip[1]:
                return img

            if noip[0] and doip[1]:
                return img[
                    :,
                    idx[1]:idx[1] + itp_fac[1] * input_sz[1],
                    :
                ]

            if doip[0] and noip[1]:
                return img[
                    idx[0]:idx[0] + itp_fac[0] * input_sz[0],
                    :,
                    :
                ]

            return img[
                idx[0]:idx[0] + itp_fac[0] * input_sz[0],
                idx[1]:idx[1] + itp_fac[1] * input_sz[1],
                :
            ]

        if noip[0] and noip[1]:
            return img[
                :,
                :,
                idx[2]:idx[2] + itp_fac[2] * input_sz[2]
            ]

        if noip[0] and doip[1]:
            return img[
                :,
                idx[1]:idx[1] + itp_fac[1] * input_sz[1],
                idx[2]:idx[2] + itp_fac[2] * input_sz[2]
            ]

        if doip[0] and noip[1]:
            return img[
                idx[0]:idx[0] + itp_fac[0] * input_sz[0],
                :,
                idx[2]:idx[2] + itp_fac[2] * input_sz[2]
            ]

        return img[
            idx[0]:idx[0] + itp_fac[0] * input_sz[0],
            idx[1]:idx[1] + itp_fac[1] * input_sz[1],
            idx[2]:idx[2] + itp_fac[2] * input_sz[2]
        ]

    return img


def fourierInterpolation(img, itp_fac, mirrorMode="none"):
    """Fourier interpolation of a 2D or 3D image."""

    img = np.asarray(img)
    itp_fac = np.atleast_1d(itp_fac).astype(int)

    if len(itp_fac) not in (1, img.ndim):
        raise ValueError(
            f"{len(itp_fac)} interpolation factors specified. "
            "Give either one for all dimensions or one per dimension!"
        )

    if np.all(itp_fac == 1):
        return img

    if len(itp_fac) == 1:
        itp_fac = np.repeat(itp_fac, img.ndim)

    noip = itp_fac == 1
    input_sz = np.array(img.shape)

    sz = input_sz.copy()
    sz -= sz % 2

    idx = (
        np.ceil(sz / 2).astype(int)
        + (itp_fac - 1) * np.floor(sz / 2).astype(int)
    )

    # MATLAB index -> Python index
    idx = idx.astype(int) - 1

    if img.ndim == 2:
        if mirrorMode == "none":
            newsz = np.round(
                itp_fac * np.array(img.shape)
            ).astype(int)
            return fInterp_2D(img, newsz)

        if mirrorMode == "lateral":
            padsize = np.array(
                img.shape,
                dtype=float
            ) / 2

            padsize[noip] = 0

            pre = np.ceil(padsize).astype(int)
            post = np.floor(padsize).astype(int)

            img = np.pad(
                img,
                (
                    (pre[0], post[0]),
                    (pre[1], post[1])
                ),
                mode="symmetric"
            )

            newsz = np.round(
                itp_fac * np.array(img.shape)
                - (itp_fac - 1)
            ).astype(int)

            img = fInterp_2D(
                img,
                newsz
            )

            return _get_valid_part(
                img,
                input_sz,
                itp_fac,
                noip,
                idx,
                mirrorMode
            )

        if mirrorMode in ("axial", "both"):
            raise ValueError(
                f"Padding '{mirrorMode}' only possible for 3D data."
            )

        raise ValueError(
            f"Unknown padding option '{mirrorMode}'."
        )

    if img.ndim == 3:
        if mirrorMode == "none":
            newsz = np.round(
                itp_fac * np.array(img.shape)
            ).astype(int)

            return _fInterp_3D(
                img,
                newsz
            )

        if mirrorMode == "lateral":
            padsize = np.array(
                [img.shape[0] / 2,
                 img.shape[1] / 2,
                 0]
            )

            padsize[noip] = 0

            pre = np.ceil(padsize).astype(int)
            post = np.floor(padsize).astype(int)

            img = np.pad(
                img,
                (
                    (pre[0], post[0]),
                    (pre[1], post[1]),
                    (pre[2], post[2])
                ),
                mode="symmetric"
            )

            newsz = np.round(
                [
                    itp_fac[0] * img.shape[0]
                    - (itp_fac[0] - 1),
                    itp_fac[1] * img.shape[1]
                    - (itp_fac[1] - 1),
                    itp_fac[2] * img.shape[2]
                ]
            ).astype(int)

            img = _fInterp_3D(
                img,
                newsz
            )

            return _get_valid_part(
                img,
                input_sz,
                itp_fac,
                noip,
                idx,
                mirrorMode
            )

        if mirrorMode == "axial":
            padsize = np.array(
                [0,
                 0,
                 img.shape[2] / 2]
            )

            padsize[noip] = 0

            pre = np.ceil(padsize).astype(int)
            post = np.floor(padsize).astype(int)

            img = np.pad(
                img,
                (
                    (pre[0], post[0]),
                    (pre[1], post[1]),
                    (pre[2], post[2])
                ),
                mode="symmetric"
            )

            newsz = np.round(
                [
                    itp_fac[0] * img.shape[0],
                    itp_fac[1] * img.shape[1],
                    itp_fac[2] * img.shape[2]
                    - (itp_fac[2] - 1)
                ]
            ).astype(int)

            img = _fInterp_3D(
                img,
                newsz
            )

            return _get_valid_part(
                img,
                input_sz,
                itp_fac,
                noip,
                idx,
                mirrorMode
            )

        if mirrorMode == "both":
            padsize = (
                np.array(img.shape, dtype=float)
                / 2
            )

            padsize[noip] = 0

            pre = np.ceil(padsize).astype(int)
            post = np.floor(padsize).astype(int)

            img = np.pad(
                img,
                (
                    (pre[0], post[0]),
                    (pre[1], post[1]),
                    (pre[2], post[2])
                ),
                mode="symmetric"
            )

            newsz = np.round(
                itp_fac * np.array(img.shape)
                - (itp_fac - 1)
            ).astype(int)

            img = _fInterp_3D(
                img,
                newsz
            )

            return _get_valid_part(
                img,
                input_sz,
                itp_fac,
                noip,
                idx,
                mirrorMode
            )

        raise ValueError(
            f"Unknown padding option '{mirrorMode}'."
        )

    raise ValueError(
        "fourierInterpolation supports only 2D or 3D data."
    )