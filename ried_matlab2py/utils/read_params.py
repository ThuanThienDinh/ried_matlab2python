# ried/Utils/read_params.py

def read_params(params, args):

    if args is None:
        args = ()

    args = tuple(args)

    if len(args) % 2 != 0:
        raise ValueError(
            "Parameters must be supplied as name/value pairs."
        )

    given = []
    parout = params

    valid_fields = set(vars(params).keys())

    for i in range(0, len(args), 2):

        name = args[i]
        value = args[i + 1]

        # MATLAB:
        #
        # if strcmp(args{i}, 'params')
        #
        if name == "params":

            if isinstance(value, dict):

                for field, field_value in value.items():

                    if field not in valid_fields:
                        raise ValueError(
                            f"Unknown parameter: {field}"
                        )

                    setattr(
                        parout,
                        field,
                        field_value
                    )

                    given.append(field)

            else:

                for field, field_value in vars(value).items():

                    if field not in valid_fields:
                        raise ValueError(
                            f"Unknown parameter: {field}"
                        )

                    setattr(
                        parout,
                        field,
                        field_value
                    )

                    given.append(field)

        # MATLAB:
        #
        # elseif strcmp(args{i}, 'varargin')
        #
        elif name == "varargin":

            parout, nested_given = read_params(
                parout,
                value
            )

            given.extend(nested_given)

        # MATLAB:
        #
        # elseif isfield(params,args{i})
        #
        elif name in valid_fields:

            setattr(
                parout,
                name,
                value
            )

            given.append(name)

        else:

            if not isinstance(name, str):
                raise TypeError(
                    f"Parameter name at position {i} "
                    f"must be a string."
                )

            raise ValueError(
                f"Unknown parameter: {name}"
            )

    return parout, given